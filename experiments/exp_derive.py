"""Can a published quantised file be re-derived, bit for bit, from the known model it claims to come from?

A compressed copy of a known model is close to it, and "close" is room for an operator. A copy that the box can
recompute itself leaves no room: every block of the file equals what the public encoder writes when it is given the
known model's own numbers, so the file holds nothing the known model does not.

For each GGUF file, each tensor is read with the format its own header states. The known model's tensor goes through
the reference encoder for that format (ggml, the library llama.cpp quantises with, loaded as a shared library; the
8-bit and plain formats are also recomputed in numpy) and the bytes are compared. The encoder is tried in each build
given: compilers differ in whether they fuse a multiply and an add, and that changes which way a tie rounds.
Calibrated files (made with an importance matrix) are not expected to match: their rounding depends on data the
publisher chose.

Usage: python exp_derive.py plan <manifest.jsonl> <base repository> ...     the published GGUF files declared as quantisations of each base
       python exp_derive.py run <manifest.jsonl> <out.jsonl>                 re-derive every file of the manifest
       python exp_derive.py files <out.jsonl> <base repository> <GGUF repository>[:<part of a file name>] ...
       python exp_derive.py explain <out.jsonl> <census.jsonl> ...   why a file did not re-derive: its weights against the
                                     declared base and its siblings, with and without a per-channel rescaling
       DERIVE_LIB=/path/build-a/bin/libggml-base.so[,/path/build-b/bin/libggml-base.so]   the encoder, one or more builds
       DERIVE_SHARD=i/n takes every n-th file; DERIVE_EVERY=1 tries every build on every tensor; DERIVE_KEEP=1 keeps downloads
"""
import ctypes, hashlib, json, os, random, re, sys, time
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from safetensors import safe_open
from lineage_check import gguf as G

LIBS = [p for p in os.environ.get("DERIVE_LIB", "").split(",") if p]
EVERY = os.environ.get("DERIVE_EVERY") == "1"
SHARD = tuple(int(v) for v in os.environ.get("DERIVE_SHARD", "0/1").split("/"))
PLAIN = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
GIVE_UP = 12                                           # quantised tensors compared, under half identical: stop reading the file
MORE = {18: "IQ3_XXS", 21: "IQ3_S", 34: "TQ1_0", 35: "TQ2_0"}      # formats the encoder writes and lineage_check does not decode
NEEDS_DATA = {16: "IQ2_XXS", 17: "IQ2_XS", 19: "IQ1_S", 22: "IQ2_S", 29: "IQ1_M"}   # written only with calibration data
_BIAS = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}
WANTED = ["Q4_K_M", "Q8_0", "Q5_K_M", "Q6_K", "Q4_0", "Q4_K_S", "Q3_K_M", "Q2_K", "F16", "Q5_K_S", "Q5_0", "IQ4_XS", "Q3_K_L", "Q3_K_S", "Q4_1", "Q5_1", "BF16", "F32"]
_LABEL = re.compile(r"(?<![A-Za-z0-9])(I?Q\d_[A-Z0-9_]*[A-Z0-9]|F16|F32|BF16)(?![A-Za-z0-9])", re.I)


class Encoder:
    """The reference encoder of one build of ggml."""

    def __init__(self, path):
        self.name, self.lib = os.path.basename(os.path.dirname(os.path.dirname(path))).replace("build-", "") or path, ctypes.CDLL(path)
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_blck_size.restype, self.lib.ggml_blck_size.argtypes = ctypes.c_int64, [ctypes.c_int]
        self.lib.ggml_type_size.restype, self.lib.ggml_type_size.argtypes = ctypes.c_size_t, [ctypes.c_int]

    def block(self, kind):
        return int(self.lib.ggml_blck_size(kind)), int(self.lib.ggml_type_size(kind))

    def __call__(self, kind, x, rows, per_row, size):
        x = np.ascontiguousarray(x, np.float32)
        out = np.zeros(size, np.uint8)
        wrote = self.lib.ggml_quantize_chunk(kind, x.ctypes.data, out.ctypes.data, 0, rows, per_row, None)
        if wrote != size:
            raise ValueError(f"the encoder wrote {wrote} bytes, the file holds {size}")
        return out


def roundf(n):                                         # C roundf: halves go away from zero
    a = np.abs(n)
    f = np.floor(a)
    return np.sign(n) * (f + np.floor(2 * (a - f)))


def q8_0(x):
    """The 8-bit format in numpy: runs of 32 numbers, one scale each."""
    b = np.ascontiguousarray(x, np.float32).reshape(-1, 32)
    d = np.abs(b).max(1, keepdims=True) / np.float32(127)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    return np.concatenate([d.astype("<f2").view(np.uint8), roundf(b * inv).astype(np.int8).view(np.uint8)], 1).reshape(-1)


def hub_name(name):
    out = G.hub_name(name)
    m = G._IN_BLOCK.match(name.replace(".bias", ".weight")) if name.endswith(".bias") else None
    return f"model.layers.{m.group(1)}.{_BIAS[m.group(2)]}.bias" if m and m.group(2) in _BIAS else out


def base_tensors(base):
    files = [f for f in list_repo_files(base) if f.endswith(".safetensors") and "/" not in f]
    cfg = json.load(open(hf_hub_download(base, "config.json")))
    opened = [safe_open(hf_hub_download(base, f), "pt") for f in files]     # torch reads bfloat16, numpy does not
    where = {k: s for s in opened for k in s.keys()}
    return cfg, lambda k: where[k].get_tensor(k).float().numpy() if k in where else None


def base_text(base):
    """The known model's tokens in id order and its chat template (None when it has none)."""
    tok = json.load(open(hf_hub_download(base, "tokenizer.json")))
    ids = dict(tok["model"].get("vocab") or {}) if isinstance(tok["model"].get("vocab"), dict) else {}
    for t in tok.get("added_tokens", []):
        ids[t["content"]] = t["id"]
    vocab = [None] * (max(ids.values()) + 1 if ids else 0)
    for t, i in ids.items():
        vocab[i] = t
    try:
        template = json.load(open(hf_hub_download(base, "tokenizer_config.json"))).get("chat_template")
    except Exception:
        template = None
    return vocab, template if template is None or isinstance(template, str) else json.dumps(template, sort_keys=True)


def sources(w):
    """The known tensor as float32, and as it reads after a pass through each 16-bit format a converter may use
    (each one only if it differs from those before it: a bfloat16 checkpoint reads the same both ways)."""
    w32 = w.astype(np.float32)
    with np.errstate(over="ignore"):
        f16 = w32.astype(np.float16).astype(np.float32)
    bits = np.ascontiguousarray(w32).view(np.uint32)
    b16 = (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)
    seen = []
    for label, x in (("float32", w32), ("through float16", f16), ("through bfloat16", b16)):
        if not any(np.array_equal(x, y, equal_nan=True) for y in seen):
            seen.append(x)
            yield label, x


class File:
    """One GGUF file next to the known model: its tensors, and the known tensor each one should come from."""

    def __init__(self, cfg, get, path, more=None):
        self.size, self.fh, self.get = os.path.getsize(path), open(path, "rb"), get
        self.types = {**G.TYPES, **({k: (v, *more(k)) for k, v in MORE.items()} if more else {})}
        self.h = G.header(self.read, self.size)
        self.arch = self.h["meta"].get("general.architecture")
        self.heads = {"q_proj": cfg.get("num_attention_heads"), "k_proj": cfg.get("num_key_value_heads", cfg.get("num_attention_heads"))} if self.arch == "llama" else {}

    def read(self, a, b):
        self.fh.seek(a)
        return self.fh.read(b - a + 1)

    def tensor(self, name):
        """(format, numbers per block, bytes per block, the file's bytes, the known tensor in the file's row order) or None."""
        kind, shape, offset = self.h["tensors"][name]
        fmt, per, bsize = self.types[kind]
        w = self.get(hub_name(name))
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if w is None or w.size != n or n % per:
            return None
        part = hub_name(name).rsplit(".", 2)[-2]
        if self.heads.get(part) and w.ndim == 2:        # llama.cpp stores Llama query and key rows with each head's halves interleaved
            hh = self.heads[part]
            w = w.reshape(hh, 2, w.shape[0] // hh // 2, w.shape[1]).swapaxes(1, 2).reshape(w.shape)
        a = self.h["data"] + offset
        return kind, fmt, per, bsize, np.frombuffer(self.read(a, a + n // per * bsize - 1), np.uint8), w, (shape[-1] if shape else 1)


def attempts(kind, fmt, w, per_row, size, encoders):
    """Every way the known tensor could have been written in this format: (label, bytes), cheapest first."""
    for label, x in sources(w):
        if fmt in PLAIN:
            yield f"{label}, cast", x.astype(PLAIN[fmt]).view(np.uint8).reshape(-1)
        elif fmt == "BF16":                              # rounded to nearest, ties to even, as the GGUF converter writes it
            bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
            yield f"{label}, cast", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").view(np.uint8).reshape(-1)
        else:
            if fmt == "Q8_0":
                yield f"{label}, numpy", q8_0(x)
            for e in encoders:
                try:
                    yield f"{label}, {e.name}", e(kind, x, x.size // per_row, per_row, size)
                except ValueError:                       # a format this build writes differently or not at all
                    continue


def reproduces(f, name, encoders):
    kind, fmt, per, bsize, raw, w, per_row = f.tensor(name)
    return any(got.size == raw.size and (got == raw).all() for _, got in attempts(kind, fmt, w, per_row, raw.size, encoders))


def one_file(cfg, get, path, encoders, text=(None, None)):
    f = File(cfg, get, path, encoders[0].block if encoders else None)
    meta = f.h["meta"]
    said = {k: (v if not isinstance(v, str) else v[:120]) for k, v in meta.items()
            if k.startswith("quantize.") or k in ("general.name", "general.quantized_by", "general.file_type", "general.quantization_version", "general.base_model.0.repo_url")}
    row = {"bytes": f.size, "architecture": f.arch, "said": said, "calibrated": any("imatrix" in k for k in meta),
           "tensors": len(f.h["tensors"]), "read": 0, "identical": 0, "blocks": 0, "blocks_identical": 0, "plain": 0, "plain_identical": 0,
           "quantised": 0, "quantised_identical": 0, "formats": {}, "by": {}, "missing": [], "unread": [], "differs": [],
           "error_file": [], "error_derived": [], "stopped_early": False, "needs_data": 0}
    encoders, how = list(encoders), {}
    for name, (kind, shape, offset) in f.h["tensors"].items():
        if kind in NEEDS_DATA:
            row["needs_data"] += 1
            continue
        if kind not in f.types:
            row["unread"].append(name)
            continue
        t = f.tensor(name)
        if t is None:
            row["missing"].append(name)
            continue
        kind, fmt, per, bsize, raw, w, per_row = t
        best, ours = ("none", -1.0), None
        for label, got in attempts(kind, fmt, w, per_row, raw.size, encoders):
            same = float((got.reshape(-1, bsize) == raw.reshape(-1, bsize)).all(1).mean()) if got.size == raw.size else 0.0
            if same == 1.0 and EVERY:
                row.setdefault("every", {})[label] = row.get("every", {}).get(label, 0) + 1
            if same > best[1]:
                best, ours = (label, same), got
            if best[1] == 1.0 and not EVERY:
                break
        best = (best[0], max(best[1], 0.0))
        blocks, plain = raw.size // bsize, fmt in PLAIN or fmt == "BF16"
        fr = row["formats"].setdefault(fmt, {"tensors": 0, "identical": 0, "blocks": 0, "blocks_identical": 0})
        for d in (fr, row):
            d["blocks"] += blocks; d["blocks_identical"] += round(best[1] * blocks); d["identical"] += best[1] == 1.0
        fr["tensors"] += 1; row["read"] += 1
        row["plain" if plain else "quantised"] += 1
        row["plain_identical" if plain else "quantised_identical"] += best[1] == 1.0
        if best[1] == 1.0:
            row["by"][best[0]] = row["by"].get(best[0], 0) + 1
            build = best[0].split(", ")[1]
            if build not in ("cast", "numpy"):
                how[name] = build
                encoders.sort(key=lambda e: e.name != build)     # the build that worked goes first for the next tensor
        elif len(row["differs"]) < 12:
            row["differs"].append([name, fmt, best[0], round(best[1], 6)])
            if not plain and ours is not None and w.ndim == 2 and fmt in G.BLOCK:
                far = lambda b: float(np.linalg.norm(G.decode(b.tobytes(), fmt) - w.reshape(-1)) / max(np.linalg.norm(w), 1e-30))
                row["error_file"].append(round(far(raw), 5)); row["error_derived"].append(round(far(ours), 5))
        if row["quantised"] >= GIVE_UP and row["quantised_identical"] < row["quantised"] / 2 and not EVERY:
            row["stopped_early"] = True
            break
    row["derived"] = bool(row["read"] and row["identical"] == len(f.h["tensors"]))
    vocab, template = text                               # the rest of what the file says about the model: tokens and chat template
    toks = f.h["tokens"]
    if toks is not None and vocab:
        n = min(len(toks), len(vocab))
        row["vocab"] = {"file": len(toks), "known": len(vocab), "compared": n, "same": sum(a == b for a, b in zip(toks[:n], vocab[:n]))}
    tmpl = meta.get("tokenizer.chat_template")
    row["template"] = ("absent in both" if tmpl is None and template is None else "same" if tmpl == template else
                       "added" if template is None else "removed" if tmpl is None else "differs")
    row["template_sha"] = hashlib.sha256(tmpl.encode()).hexdigest()[:12] if isinstance(tmpl, str) else None
    builds = sorted(set(how.values()))
    if row["derived"] and len(builds) > 1:                # does one build write the whole file?
        for b in sorted(builds, key=lambda b: -list(how.values()).count(b)):
            if all(reproduces(f, n, [e for e in encoders if e.name == b]) for n, v in how.items() if v != b):
                builds = [b]
                break
    row["builds"] = builds
    return row


def label(file):
    found = _LABEL.findall(file.rsplit("/", 1)[-1])
    return found[-1].upper() if found else "?"


def plan(out, *bases):
    """The published GGUF files declared as quantisations of each base: the most downloaded repositories and a random
    draw of the rest. DERIVE_TOP, DERIVE_DRAW repositories per base; DERIVE_PER files per repository."""
    api, rng = HfApi(), random.Random(0)
    top, draw, per = (int(os.environ.get(k, v)) for k, v in (("DERIVE_TOP", 20), ("DERIVE_DRAW", 20), ("DERIVE_PER", 12)))
    total = 0
    with open(out, "a") as fh:
        for base in bases:
            repos = [m for m in api.list_models(filter=f"base_model:quantized:{base}", sort="downloads", limit=1000) if "gguf" in (m.tags or [])]
            rest = repos[top:]
            chosen = repos[:top] + rng.sample(rest, min(draw, len(rest)))
            size = files = 0
            for i, m in enumerate(chosen):
                try:
                    info = api.model_info(m.id, files_metadata=True)
                except Exception as err:
                    print(f"  {m.id}: {type(err).__name__}", flush=True)
                    continue
                names = [(s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(".gguf") and "mmproj" not in s.rfilename.lower()
                         and not G._SHARD.search(s.rfilename)]
                names.sort(key=lambda t: (WANTED.index(label(t[0])) if label(t[0]) in WANTED else len(WANTED), t[0]))
                for name, sz in names[:per]:
                    fh.write(json.dumps({"base": base, "repo": m.id, "downloads": m.downloads, "created": str(info.created_at)[:10], "drawn": i >= top,
                                         "published": len(repos), "file": name, "format": label(name), "bytes": sz}) + "\n")
                    size += sz; files += 1
            total += size
            print(f"{base}: {len(repos)} GGUF repositories declared, {len(chosen)} taken, {files} files, {size / 1e9:.1f} GB", flush=True)
    print(f"total {total / 1e9:.1f} GB")


def run(manifest, out):
    encoders = [Encoder(p) for p in LIBS]
    rows = [json.loads(l) for l in open(manifest)]
    done = {(r["repo"], r["file"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    base, cfg, get, text = None, None, None, (None, None)
    for i, m in enumerate(rows):
        if i % SHARD[1] != SHARD[0] or (m["repo"], m["file"]) in done:
            continue
        t0 = time.time()
        try:
            if m["base"] != base:
                base = m["base"]
                (cfg, get), text = base_tensors(base), base_text(base)
            path = hf_hub_download(m["repo"], m["file"])
            row = {**m, **one_file(cfg, get, path, encoders, text)}
            if os.environ.get("DERIVE_KEEP") != "1":
                os.remove(os.path.realpath(path))        # the file is large and is not needed again
        except Exception as err:
            row = {**m, "error": f"{type(err).__name__}: {str(err)[:200]}"}
        row["seconds"] = round(time.time() - t0, 1)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if "error" in row:
            print(f"{m['repo']}/{m['file']}: {row['error']}", flush=True)
        else:
            print(f"{m['repo']}/{m['file']}: {'DERIVED ' + '+'.join(row['builds']) if row['derived'] else 'not derived'}  tensors {row['identical']}/{row['read']}"
                  f"{' (stopped early)' if row['stopped_early'] else ''}  blocks {row['blocks_identical']}/{row['blocks']}  plain {row['plain_identical']}/{row['plain']}"
                  f"  {'calibrated' if row['calibrated'] else ''}  vocab {row.get('vocab', {}).get('same')}/{row.get('vocab', {}).get('compared')}  template {row['template']}"
                  f"  {row['differs'][:1] if row['differs'] else ''}  {row['seconds']} s", flush=True)


def files(out, base, *specs):
    manifest = out + ".manifest"
    with open(manifest, "w") as fh:
        for spec in specs:
            repo, _, want = spec.partition(":")
            for name in sorted(f for f in list_repo_files(repo) if f.endswith(".gguf") and want.lower() in f.lower() and "mmproj" not in f.lower()):
                fh.write(json.dumps({"base": base, "repo": repo, "file": name, "format": label(name)}) + "\n")
    run(manifest, out)


# ------------------------------------------------------------------ why a file did not re-derive
SIBLINGS = {}
for group in (["HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-135M-Instruct"], ["HuggingFaceTB/SmolLM-360M", "HuggingFaceTB/SmolLM-360M-Instruct"],
              ["HuggingFaceTB/SmolLM-1.7B", "HuggingFaceTB/SmolLM-1.7B-Instruct"], ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"],
              ["HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"], ["HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"],
              ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-0.5B", "Qwen/Qwen2.5-Coder-0.5B-Instruct"],
              ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-1.5B-Instruct"],
              ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama_v1.1", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"]):
    for b in group:
        SIBLINGS[b] = [o for o in group if o != b]


def _weights(f, name):
    kind, shape, offset = f.h["tensors"][name]
    fmt, per, bsize = f.types[kind]
    n = int(np.prod(shape, dtype=np.int64))
    a = f.h["data"] + offset
    return fmt, G.decode(f.read(a, a + n // per * bsize - 1), fmt).reshape(shape) if fmt in G.BLOCK or fmt in PLAIN or fmt == "BF16" else None


def explain_one(path, bases, encoders):
    """For each candidate base: how far the file's weights are from it, against how far an honest encoding of it in the
    same format would be, plainly and after undoing a per-channel scale read off the norm weights."""
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    out = {}
    for base in bases:
        try:
            cfg, get = base_tensors(base)
        except Exception as err:
            out[base] = {"error": type(err).__name__}
            continue
        f = File(cfg, get, path, encoders[0].block if encoders else None)
        n_layers = 1 + max(int(m.group(1)) for m in map(G._IN_BLOCK.match, f.h["tensors"]) if m)
        ratios, rescaled, shapes_ok, where = [], [], True, []
        for L in sorted({0, n_layers // 2, n_layers - 1}):
            for mat in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
                name = f"blk.{L}.{mat}.weight"
                if name not in f.h["tensors"]:
                    continue
                t = f.tensor(name)
                if t is None:
                    shapes_ok = False
                    continue
                kind, fmt, per, bsize, raw, w, per_row = t
                if fmt not in G.BLOCK and fmt not in PLAIN and fmt != "BF16":
                    continue
                got = G.decode(raw.tobytes(), fmt).reshape(w.shape)
                honest = next(iter(attempts(kind, fmt, w, per_row, raw.size, encoders)), (None, None))[1]
                e_d = rel(G.decode(honest.tobytes(), fmt).reshape(w.shape), w) if honest is not None else 0.0
                e_f = rel(got, w)
                ratios.append(e_f / max(e_d, 1e-4))
                where.append(f"{L}.{mat}")
                if mat in ("attn_q", "attn_k") and f"blk.{L}.attn_norm.weight" in f.h["tensors"]:
                    _, norm_f = _weights(f, f"blk.{L}.attn_norm.weight")
                    norm_b = get(hub_name(f"blk.{L}.attn_norm.weight"))
                    if norm_f is not None and norm_b is not None and norm_b.shape == norm_f.shape:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            sc = np.where(norm_f != 0, norm_b / norm_f, 1.0)
                        rescaled.append(rel(got, w * sc[None, :]) / max(e_d, 1e-4))
        out[base] = {"shapes": shapes_ok, "matrices": len(ratios), "median_ratio": round(float(np.median(ratios)), 3) if ratios else None,
                     "max_ratio": round(float(np.max(ratios)), 3) if ratios else None,
                     "rescaled_ratio": round(float(np.median(rescaled)), 3) if rescaled else None,
                     "edited": [w for w, r in zip(where, ratios) if r > 3]}
    return out


def explain(out, *census):
    encoders = [Encoder(p) for p in LIBS[:1]]
    rows = [json.loads(l) for c in census for l in open(c)]
    todo = [r for r in rows if "error" not in r and not r.get("derived") and not r.get("calibrated")]
    done = {(r["repo"], r["file"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    for i, r in enumerate(todo):
        if i % SHARD[1] != SHARD[0] or (r["repo"], r["file"]) in done:
            continue
        t0 = time.time()
        try:
            path = hf_hub_download(r["repo"], r["file"])
            seen = explain_one(path, [r["base"]] + SIBLINGS.get(r["base"], []), encoders)
            if os.environ.get("DERIVE_KEEP") != "1":
                os.remove(os.path.realpath(path))
            near = {b: v for b, v in seen.items() if v.get("median_ratio") is not None and v["shapes"]}
            best = min(near, key=lambda b: near[b]["median_ratio"]) if near else None
            if best is None:
                verdict = "other shapes"
            elif near[best]["median_ratio"] <= 1.5 and near[best]["max_ratio"] > 3:
                verdict = "edited in places" if best == r["base"] else "a sibling's weights, edited in places"
            elif near[best]["median_ratio"] <= 1.5:
                verdict = "same weights, other rounding" if best == r["base"] else "a sibling's weights"
            elif min((v["rescaled_ratio"] or 1e9) for v in near.values()) <= 1.5:
                verdict = "rescaled per channel"
            else:
                verdict = "different weights"
            row = {"repo": r["repo"], "file": r["file"], "base": r["base"], "closest": best, "verdict": verdict, "bases": seen}
        except Exception as err:
            row = {"repo": r["repo"], "file": r["file"], "base": r["base"], "error": f"{type(err).__name__}: {str(err)[:200]}"}
        row["seconds"] = round(time.time() - t0, 1)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{r['repo']}/{r['file']}: {row.get('verdict', row.get('error'))}  closest {row.get('closest')}  "
              f"{ {b.split('/')[1]: (v.get('median_ratio'), v.get('rescaled_ratio')) for b, v in row.get('bases', {}).items()} }  {row['seconds']} s", flush=True)


if __name__ == "__main__":
    {"plan": plan, "run": run, "files": files, "explain": explain}[sys.argv[1]](*sys.argv[2:])
