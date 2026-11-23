"""GGUF files: the header, the block formats, and a quantised GGUF copy going through the traffic light."""
import struct

import numpy as np
import pytest

from conftest import tiny_model
from lineage_check import edits, gguf, light, remote
from test_light import KNOWN, LISTED, NONCE, OTHER, REGISTRY, WATCH, WORDS

F32, Q8_0 = 0, 8


# ------------------------------------------------------------------ writing a file
def _text(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def q8_0(x):
    """Encode numbers as Q8_0: blocks of 32, one half-precision scale and 32 signed bytes each."""
    b = np.asarray(x, np.float32).reshape(-1, 32)
    d = (np.abs(b).max(1, keepdims=True) / 127).astype("<f2")
    q = np.round(b / np.maximum(d.astype(np.float32), 1e-30)).clip(-127, 127).astype(np.int8)
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), q.view(np.uint8)], axis=1).tobytes()


def write_gguf(path, tensors, tokens, alignment=32):
    """tensors: {name: array}. Matrices are stored as Q8_0, everything else as F32."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 3)
    head += _text("general.architecture") + struct.pack("<I", 8) + _text("llama")
    head += _text("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", alignment)
    head += _text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(_text, tokens))
    body, offset = b"", 0
    for name, x in tensors.items():
        kind = Q8_0 if x.ndim == 2 else F32
        raw = q8_0(x) if kind == Q8_0 else np.asarray(x, "<f4").tobytes()
        raw += b"\0" * (-len(raw) % alignment)
        head += _text(name) + struct.pack("<I", x.ndim) + b"".join(struct.pack("<Q", n) for n in reversed(x.shape))
        head += struct.pack("<IQ", kind, offset)
        body, offset = body + raw, offset + len(raw)
    head += b"\0" * (-len(head) % alignment)
    with open(path, "wb") as f:
        f.write(head + body)
    return path


def as_gguf(model, tmp_path, name="m.gguf"):
    """The tiny model as a GGUF file, under the names llama.cpp gives its tensors."""
    t = {"token_embd.weight": model["embed"].numpy()}
    for i, l in enumerate(model["layers"]):
        for short, key in (("attn_q", "q"), ("attn_k", "k"), ("attn_v", "v"), ("attn_output", "o"),
                           ("ffn_gate", "gate"), ("ffn_up", "up"), ("ffn_down", "down"), ("attn_norm", "g1"), ("ffn_norm", "g2")):
            t[f"blk.{i}.{short}.weight"] = l[key].numpy()
    tokens = [f"<{i}>" for i in range(300)] + WORDS
    return remote.LocalGGUF([write_gguf(str(tmp_path / name), t, tokens)])


# ------------------------------------------------------------------ the header
def test_the_header_gives_tokens_shapes_and_where_the_data_starts(tmp_path):
    x = np.arange(64, dtype=np.float32).reshape(2, 32)
    path = write_gguf(str(tmp_path / "a.gguf"), {"blk.0.attn_q.weight": x, "blk.0.attn_norm.weight": x[0]}, ["a", "b"])
    raw = open(path, "rb").read()
    fetched = []
    h = gguf.header(lambda a, b: fetched.append((a, b)) or raw[a:b + 1], len(raw), first=16)
    assert len(fetched) > 3                                        # the header was read in several pieces
    assert h["tokens"] == ["a", "b"] and h["meta"]["general.architecture"] == "llama"
    assert h["tensors"]["blk.0.attn_q.weight"][:2] == [Q8_0, [2, 32]] and h["data"] % 32 == 0
    index = gguf.entries(h, path)
    assert set(index) == {"model.layers.0.self_attn.q_proj.weight", "model.layers.0.input_layernorm.weight"}
    f, fmt, shape, a, b = index["model.layers.0.input_layernorm.weight"]
    assert np.array_equal(gguf.decode(raw[a:b], fmt[len(gguf.PREFIX):], shape), x[0])


def test_something_that_is_not_gguf_is_refused():
    with pytest.raises(ValueError):
        gguf.header(lambda a, b: b"NOPE"[a:b + 1], 4)


def test_the_file_most_people_run_is_chosen_with_all_its_shards():
    files = ["README.md", "m-Q8_0.gguf", "mmproj-m-f16.gguf", "Q4_K_M/m-Q4_K_M-00002-of-00002.gguf", "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf"]
    assert gguf.choose(files) == ["Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", "Q4_K_M/m-Q4_K_M-00002-of-00002.gguf"]
    assert gguf.choose(files, "q8_0") == ["m-Q8_0.gguf"]
    assert gguf.choose(["model.safetensors"]) == []


# ------------------------------------------------------------------ the block formats, against the format written out longhand
def _half(b, i):
    return float(np.frombuffer(bytes(b[i:i + 2]), "<f2")[0])


def _signed(v):
    return v - 256 if v > 127 else v


def _k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def ref_q8_0(b):
    return [_half(b, 0) * _signed(b[2 + j]) for j in range(32)]


def ref_q4_0(b):
    d, q = _half(b, 0), b[2:18]
    return [((q[j] & 15) - 8) * d for j in range(16)] + [((q[j] >> 4) - 8) * d for j in range(16)]


def ref_q4_1(b):
    d, m, q = _half(b, 0), _half(b, 2), b[4:20]
    return [(q[j] & 15) * d + m for j in range(16)] + [(q[j] >> 4) * d + m for j in range(16)]


def _five(high, q, j):
    h = int.from_bytes(bytes(high), "little")
    return (q[j] & 15) | (((h >> j) << 4) & 16), (q[j] >> 4) | ((h >> (j + 12)) & 16)


def ref_q5_0(b):
    d, pairs = _half(b, 0), [_five(b[2:6], b[6:22], j) for j in range(16)]
    return [(x - 16) * d for x, _ in pairs] + [(y - 16) * d for _, y in pairs]


def ref_q5_1(b):
    d, m, pairs = _half(b, 0), _half(b, 2), [_five(b[4:8], b[8:24], j) for j in range(16)]
    return [x * d + m for x, _ in pairs] + [y * d + m for _, y in pairs]


def ref_q2_k(b):
    scales, q, d, dmin, y, i = b[0:16], b[16:80], _half(b, 80), _half(b, 82), [], 0
    for half in (0, 32):
        for shift in (0, 2, 4, 6):
            for off in (0, 16):
                dl, ml = d * (scales[i] & 15), dmin * (scales[i] >> 4)
                y += [dl * ((q[half + off + l] >> shift) & 3) - ml for l in range(16)]
                i += 1
    return y


def ref_q3_k(b):
    mask, q, d, y, i, m = b[0:32], b[32:96], _half(b, 108), [], 0, 1
    a0, a1, t = struct.unpack("<3I", bytes(b[96:108]))
    k1, k2 = 0x03030303, 0x0f0f0f0f
    scales = struct.pack("<4I", (a0 & k2) | (((t >> 0) & k1) << 4), (a1 & k2) | (((t >> 2) & k1) << 4),
                         ((a0 >> 4) & k2) | (((t >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((t >> 6) & k1) << 4))
    for half in (0, 32):
        for shift in (0, 2, 4, 6):
            for off in (0, 16):
                dl = d * (scales[i] - 32)
                y += [dl * (((q[half + off + l] >> shift) & 3) - (0 if mask[off + l] & m else 4)) for l in range(16)]
                i += 1
            m <<= 1
    return y


def ref_q4_k(b):
    d, dmin, scales, q, y = _half(b, 0), _half(b, 2), b[4:16], b[16:144], []
    for c in range(4):
        (s1, m1), (s2, m2) = _k4(2 * c, scales), _k4(2 * c + 1, scales)
        y += [d * s1 * (q[32 * c + l] & 15) - dmin * m1 for l in range(32)]
        y += [d * s2 * (q[32 * c + l] >> 4) - dmin * m2 for l in range(32)]
    return y


def ref_q5_k(b):
    d, dmin, scales, high, q, y = _half(b, 0), _half(b, 2), b[4:16], b[16:48], b[48:176], []
    for c in range(4):
        (s1, m1), (s2, m2) = _k4(2 * c, scales), _k4(2 * c + 1, scales)
        y += [d * s1 * ((q[32 * c + l] & 15) + (16 if high[l] & (1 << 2 * c) else 0)) - dmin * m1 for l in range(32)]
        y += [d * s2 * ((q[32 * c + l] >> 4) + (16 if high[l] & (2 << 2 * c) else 0)) - dmin * m2 for l in range(32)]
    return y


def ref_q6_k(b):
    d, y = _half(b, 208), [0.0] * 256
    for o in (0, 1):
        low, high, sc = b[64 * o:64 * o + 64], b[128 + 32 * o:160 + 32 * o], [_signed(v) for v in b[192 + 8 * o:200 + 8 * o]]
        for l in range(32):
            i = l // 16
            y[128 * o + l] = d * sc[i] * (((low[l] & 15) | ((high[l] & 3) << 4)) - 32)
            y[128 * o + l + 32] = d * sc[i + 2] * (((low[l + 32] & 15) | (((high[l] >> 2) & 3) << 4)) - 32)
            y[128 * o + l + 64] = d * sc[i + 4] * (((low[l] >> 4) | (((high[l] >> 4) & 3) << 4)) - 32)
            y[128 * o + l + 96] = d * sc[i + 6] * (((low[l + 32] >> 4) | (((high[l] >> 6) & 3) << 4)) - 32)
    return y


LEVELS = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def ref_iq4_nl(b):
    d, q = _half(b, 0), b[2:18]
    return [d * LEVELS[q[j] & 15] for j in range(16)] + [d * LEVELS[q[j] >> 4] for j in range(16)]


def ref_iq4_xs(b):
    d, high, low, y = _half(b, 0), b[2] | (b[3] << 8), b[4:8], []
    for i in range(8):
        dl = d * ((((low[i // 2] >> 4 * (i % 2)) & 15) | (((high >> 2 * i) & 3) << 4)) - 32)
        q = b[8 + 16 * i:24 + 16 * i]
        y += [dl * LEVELS[q[j] & 15] for j in range(16)] + [dl * LEVELS[q[j] >> 4] for j in range(16)]
    return y


HALVES = {"Q8_0": [0], "Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q2_K": [80, 82], "Q3_K": [108],
          "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208], "IQ4_NL": [0], "IQ4_XS": [0]}


@pytest.mark.parametrize("fmt", sorted(HALVES))
def test_a_block_decodes_to_what_the_format_says(fmt):
    rng = np.random.default_rng(1)
    per, size = gguf.BLOCK[fmt]
    blocks = rng.integers(0, 256, (20, size), dtype=np.uint8)
    for at in HALVES[fmt]:                                         # the scales: finite half-precision numbers
        blocks[:, at:at + 2] = rng.normal(0, 0.01, (20, 1)).astype("<f2").view(np.uint8).reshape(20, 2)
    want = np.array([globals()["ref_" + fmt.lower()]([int(v) for v in b]) for b in blocks])
    got = gguf.decode(blocks.tobytes(), fmt).reshape(20, per)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7)


def test_rows_are_cut_out_of_blocks_that_run_across_row_ends(tmp_path):
    x = np.random.default_rng(2).normal(size=(40, 48)).astype(np.float32)       # 48 columns: rows do not end on blocks of 32
    src = remote.LocalGGUF([write_gguf(str(tmp_path / "r.gguf"), {"token_embd.weight": x}, ["t"])])
    whole = src.get("model.embed_tokens.weight")
    assert whole.shape == (40, 48) and np.abs(whole - x).max() < 0.02
    assert np.array_equal(src.rows("model.embed_tokens.weight", 7, 19), whole[7:19])


# ------------------------------------------------------------------ through the traffic light
def answer(model, tmp_path, **kw):
    r = light.light(as_gguf(model, tmp_path), WATCH, REGISTRY, NONCE, **kw)
    return r["light"], r["matched"], r["finding"]


def test_a_quantised_gguf_copy_of_a_listed_model_is_red(tmp_path):
    assert answer(LISTED, tmp_path) == ("red", "listed", "listed")


def test_a_quantised_gguf_copy_of_a_known_model_is_green_and_was_checked_for_a_cut(tmp_path):
    assert answer(KNOWN, tmp_path) == ("green", "known", "known")


def test_a_cut_shows_through_gguf_quantisation(tmp_path):
    assert answer(edits.rank1_edit(KNOWN), tmp_path) == ("red", "known", "cut")


def test_an_unrelated_gguf_model_is_amber(tmp_path):
    assert answer(OTHER, tmp_path) == ("amber", None, "none")
