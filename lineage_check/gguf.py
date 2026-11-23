"""Read GGUF model files: the header, the token list, and tensors in the common block formats.

A GGUF file is a header (settings, the tokenizer, one entry per tensor) followed by the tensors. Most public GGUF
files are quantised: numbers are stored in blocks of 32 or 256 that share a scale. `decode` turns such a block back
into plain numbers, so the rest of the reader treats a GGUF tensor like any other.

Decoded: F32, F16, BF16, F64, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS. Tensors in
other formats (the codebook formats IQ1 to IQ3 and the ternary ones) are left out of the index, and a model with
nothing left to read is Amber.
"""
from __future__ import annotations

import re
import struct

import numpy as np

# format id -> (name, numbers per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
         8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136), 28: ("F64", 1, 8), 30: ("BF16", 1, 2)}
BLOCK = {name: (per, size) for name, per, size in TYPES.values()}
PREFIX = "GGUF_"                                   # how these formats are named in a source's index

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
PREFER = ("Q4_K_M", "Q4_K_S", "Q4_K", "Q5_K_M", "Q5_K_S", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q3_K_M", "Q3_K_L", "Q3_K_S",
          "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "BF16", "F16", "F32")

_BLOCK_NAMES = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
                "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
_TOP_NAMES = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight",
              "output_norm.weight": "model.norm.weight"}
_IN_BLOCK = re.compile(r"blk\.(\d+)\.(\w+)\.weight$")


# ------------------------------------------------------------------ the header
class _Cursor:
    """A position in the start of a file. Fetches more of the file when a read runs past what it holds."""

    def __init__(self, read, size, first):
        self.read, self.size, self.buf, self.pos, self.step = read, size, bytearray(), 0, first

    def take(self, n):
        end = self.pos + n
        if end > self.size:
            raise ValueError("the GGUF header runs past the end of the file")
        while end > len(self.buf):
            have = len(self.buf)
            self.buf += self.read(have, min(self.size, max(end, have + self.step)) - 1)
            self.step *= 2
        out = bytes(self.buf[self.pos:end])
        self.pos = end
        return out

    def num(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def text(self):
        return self.take(self.num("<Q")).decode("utf-8", "replace")


def header(read, size, first=1 << 20):
    """Parse the header of a GGUF file. `read(a, b)` returns bytes a to b of the file, both included.
    Returns {"meta": the scalar settings, "tokens": the token list or None,
             "tensors": {name: [format id, shape, offset]}, "data": where the tensor data starts}."""
    c = _Cursor(read, size, first)
    if c.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = c.num("<I")
    if version not in (2, 3):
        raise ValueError(f"GGUF version {version} is not supported")
    n_tensors, n_settings = c.num("<Q"), c.num("<Q")
    meta, tokens = {}, None
    for _ in range(n_settings):
        key, kind = c.text(), c.num("<I")
        if kind == _STRING:
            meta[key] = c.text()
        elif kind in _SCALAR:
            meta[key] = c.num(_SCALAR[kind])
        elif kind == _ARRAY:
            item, count = c.num("<I"), c.num("<Q")
            if item == _STRING and key == "tokenizer.ggml.tokens":
                tokens = [c.text() for _ in range(count)]
            elif item == _STRING:
                for _ in range(count):
                    c.take(c.num("<Q"))
            elif item in _SCALAR:
                c.take(count * struct.calcsize(_SCALAR[item]))
            else:
                raise ValueError(f"GGUF setting {key}: arrays of type {item} are not supported")
        else:
            raise ValueError(f"GGUF setting {key}: type {kind} is not supported")
    tensors = {}
    for _ in range(n_tensors):
        name = c.text()
        dims = [c.num("<Q") for _ in range(c.num("<I"))]
        kind, offset = c.num("<I"), c.num("<Q")
        tensors[name] = [kind, list(reversed(dims)), offset]      # GGUF lists the last axis first
    align = int(meta.get("general.alignment", 32))
    return {"meta": meta, "tokens": tokens, "tensors": tensors, "data": (c.pos + align - 1) // align * align}


def hub_name(name):
    """The name a tensor has in a Hub checkpoint, where the two are known to correspond. Tables are matched by
    content and do not need this; the cut test compares matrices under the same name, and does."""
    if name in _TOP_NAMES:
        return _TOP_NAMES[name]
    m = _IN_BLOCK.match(name)
    if m and m.group(2) in _BLOCK_NAMES:
        return f"model.layers.{m.group(1)}.{_BLOCK_NAMES[m.group(2)]}.weight"
    return name


def entries(h, file):
    """Index entries for the tensors of a parsed header that can be decoded: {name: (file, format, shape, start, end)}."""
    out = {}
    for name, (kind, shape, offset) in h["tensors"].items():
        if kind not in TYPES:
            continue
        fmt, per, size = TYPES[kind]
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if n % per:
            continue
        a = h["data"] + offset
        out[hub_name(name)] = (file, PREFIX + fmt, tuple(shape), a, a + n // per * size)
    return out


def choose(files, want=None):
    """The GGUF file to read from a repository's file list, with every shard of it, or [] if there is none.
    `want` is part of a file name ("Q8_0"). Without it the 4-bit file most people run comes first, then finer ones."""
    sets = {}
    for f in files:
        low = f.lower()
        if low.endswith(".gguf") and "mmproj" not in low and "imatrix" not in low:
            sets.setdefault(_SHARD.sub("", f), []).append(f)
    if want:
        sets = {k: v for k, v in sets.items() if want.lower() in k.lower()}
    if not sets:
        return []
    def rank(key):                                  # a preferred format, then any plain Q format, then the rest
        tags = [i for i, t in enumerate(PREFER) if re.search(rf"(?<![A-Za-z0-9]){t}(?![A-Za-z0-9])", key, re.I)]
        return (min(tags, default=len(PREFER)), not re.search(r"(?<![A-Za-z0-9])Q[2-8]_", key, re.I), key)
    return sorted(sets[min(sets, key=rank)])


# ------------------------------------------------------------------ block formats
def _half(b):
    """Two bytes per block as a float32 column."""
    return np.ascontiguousarray(b).view("<f2").astype(np.float32)


def _nibbles(q):
    """Each byte's low four bits, then each byte's high four bits."""
    return np.concatenate([q & 15, q >> 4], axis=-1)


def _scales_k4(s):
    """The eight 6-bit scales and eight 6-bit minimums that Q4_K and Q5_K pack into twelve bytes."""
    sc = np.concatenate([s[:, 0:4] & 63, (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)], axis=1)
    mn = np.concatenate([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], axis=1)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q8_0(b):
    return _half(b[:, 0:2]) * b[:, 2:34].astype(np.int8)


def _q4_0(b):
    return _half(b[:, 0:2]) * (_nibbles(b[:, 2:18]).astype(np.float32) - 8)


def _q4_1(b):
    return _half(b[:, 0:2]) * _nibbles(b[:, 4:20]) + _half(b[:, 2:4])


def _five_bits(high, q):
    bit = (np.ascontiguousarray(high).view("<u4") >> np.arange(32, dtype=np.uint32)) & 1
    return (_nibbles(q) | (bit << 4).astype(np.uint8)).astype(np.float32)


def _q5_0(b):
    return _half(b[:, 0:2]) * (_five_bits(b[:, 2:6], b[:, 6:22]) - 16)


def _q5_1(b):
    return _half(b[:, 0:2]) * _five_bits(b[:, 4:8], b[:, 8:24]) + _half(b[:, 2:4])


def _two_bits(q):
    """256 two-bit numbers from 64 bytes, in the order Q2_K and Q3_K store them: (blocks, 16 groups, 16 numbers)."""
    q = q.reshape(-1, 2, 1, 2, 16)                                 # half, -, which sixteen bytes, byte
    shift = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1, 1)
    return ((q >> shift) & 3).reshape(-1, 16, 16)


def _q2_k(b):
    sc, q = b[:, 0:16], _two_bits(b[:, 16:80]).astype(np.float32)
    d, dmin = _half(b[:, 80:82]), _half(b[:, 82:84])
    return ((d * (sc & 15))[:, :, None] * q - (dmin * (sc >> 4))[:, :, None]).reshape(-1, 256)


def _q3_k(b):
    high, q, d = b[:, 0:32], _two_bits(b[:, 32:96]).astype(np.int8), _half(b[:, 108:110])
    aux = np.ascontiguousarray(b[:, 96:108]).view("<u4")
    k1, k2, t = np.uint32(0x03030303), np.uint32(0x0f0f0f0f), aux[:, 2]
    packed = np.stack([(aux[:, 0] & k2) | (((t >> 0) & k1) << 4), (aux[:, 1] & k2) | (((t >> 2) & k1) << 4),
                       ((aux[:, 0] >> 4) & k2) | (((t >> 4) & k1) << 4), ((aux[:, 1] >> 4) & k2) | (((t >> 6) & k1) << 4)], axis=1)
    sc = np.ascontiguousarray(packed.astype("<u4")).view(np.uint8).astype(np.float32) - 32      # sixteen 6-bit scales
    bit = np.arange(8, dtype=np.uint8).reshape(1, 2, 4, 1, 1)      # the mask bit for (half, shift)
    set_ = ((high.reshape(-1, 1, 1, 2, 16) >> bit) & 1).reshape(-1, 16, 16).astype(np.int8)
    return ((d * sc)[:, :, None] * (q - 4 * (1 - set_))).reshape(-1, 256)


def _q4_k(b):
    sc, mn = _scales_k4(b[:, 4:16])
    d, dmin = _half(b[:, 0:2]), _half(b[:, 2:4])
    q = b[:, 16:144].reshape(-1, 4, 32)
    q = np.stack([q & 15, q >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
    return ((d * sc)[:, :, None] * q - (dmin * mn)[:, :, None]).reshape(-1, 256)


def _q5_k(b):
    sc, mn = _scales_k4(b[:, 4:16])
    d, dmin = _half(b[:, 0:2]), _half(b[:, 2:4])
    q = b[:, 48:176].reshape(-1, 4, 32)
    q = np.stack([q & 15, q >> 4], axis=2).reshape(-1, 8, 32)
    fifth = (b[:, 16:48].reshape(-1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1
    q = (q | (fifth << 4)).astype(np.float32)
    return ((d * sc)[:, :, None] * q - (dmin * mn)[:, :, None]).reshape(-1, 256)


def _q6_k(b):
    low = b[:, 0:128].reshape(-1, 2, 2, 32)                        # half, which 32 bytes, byte
    low = np.concatenate([low & 15, low >> 4], axis=2)             # half, group of 32 numbers, number
    high = (b[:, 128:192].reshape(-1, 2, 1, 32) >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1)) & 3
    q = (low | (high << 4)).astype(np.float32).reshape(-1, 256) - 32
    sc = np.repeat(b[:, 192:208].astype(np.int8).astype(np.float32), 16, axis=1)
    return _half(b[:, 208:210]) * sc * q


_IQ4 = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)   # the 16 levels of IQ4


def _iq4_nl(b):
    return _half(b[:, 0:2]) * _IQ4[_nibbles(b[:, 2:18])]


def _iq4_xs(b):
    high, low, i = np.ascontiguousarray(b[:, 2:4]).view("<u2").astype(np.uint32), b[:, 4:8], np.arange(8)
    sc = ((low[:, i // 2] >> (4 * (i % 2)).astype(np.uint8)) & 15) | (((high >> (2 * i).astype(np.uint32)) & 3) << 4)
    q = b[:, 8:136].reshape(-1, 8, 16)
    return ((_half(b[:, 0:2]) * (sc.astype(np.float32) - 32))[:, :, None] * _IQ4[np.concatenate([q & 15, q >> 4], axis=2)]).reshape(-1, 256)


_DECODE = {"IQ4_NL": _iq4_nl, "IQ4_XS": _iq4_xs, "Q8_0": _q8_0, "Q4_0": _q4_0, "Q4_1": _q4_1, "Q5_0": _q5_0, "Q5_1": _q5_1,
           "Q2_K": _q2_k, "Q3_K": _q3_k, "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k}


def decode(raw, fmt, shape=None, chunk=1 << 15):
    """The bytes of a tensor, or of whole blocks of one, as float32. `fmt` is a name from TYPES."""
    if fmt == "F32":
        out = np.frombuffer(raw, "<f4").copy()                     # a copy the caller may write to
    elif fmt == "F16":
        out = np.frombuffer(raw, "<f2").astype(np.float32)
    elif fmt == "F64":
        out = np.frombuffer(raw, "<f8").astype(np.float32)
    elif fmt == "BF16":
        out = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        per, size = BLOCK[fmt]
        blocks = np.frombuffer(raw, np.uint8).reshape(-1, size)
        out = np.empty((len(blocks), per), np.float32)
        for a in range(0, len(blocks), chunk):                     # in pieces: the intermediate arrays are large
            out[a:a + chunk] = _DECODE[fmt](blocks[a:a + chunk])
        out = out.reshape(-1)
    return out if shape is None else out.reshape(shape)


def row_span(fmt, width, lo, hi):
    """Where rows lo to hi of a matrix with `width` columns sit: (first byte, last byte + 1, numbers to drop at the
    front), counted from the start of the tensor. Blocks run across row ends, so whole blocks are read and trimmed."""
    per, size = BLOCK[fmt]
    first, last = lo * width // per, -(-hi * width // per)
    return first * size, last * size, lo * width - first * per
