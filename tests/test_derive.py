"""Exact identity: a GGUF file written from the known model's own tensors re-derives; one changed byte does not."""
import os
import struct

import numpy as np
import pytest

from lineage_check import derive as D


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, tokens=None, arch="llama"):
    """A minimal GGUF v3 file: (name, ggml type id, shape, bytes) per tensor, and optionally a token list."""
    kv = [("general.architecture", 8, arch)] + ([("tokenizer.ggml.tokens", 9, tokens)] if tokens else [])
    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        out += _s(k) + struct.pack("<I", t)
        out += _s(v) if t == 8 else struct.pack("<IQ", 8, len(v)) + b"".join(_s(x) for x in v)
    offset, data = 0, bytearray()
    for name, kind, shape, raw in tensors:
        out += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape)) + struct.pack("<IQ", kind, offset)
        pad = (-len(raw)) % 32
        data += raw + b"\0" * pad
        offset += len(raw) + pad
    out += b"\0" * ((-len(out)) % 32) + data
    open(path, "wb").write(bytes(out))


def known(seed=0):
    rng = np.random.default_rng(seed)
    t = {"model.layers.0.input_layernorm.weight": (1 + 0.1 * rng.standard_normal(64)).astype(np.float32),
         "model.layers.0.mlp.down_proj.weight": (0.05 * rng.standard_normal((8, 64))).astype(np.float32)}
    return D.Known(t.get, {"num_attention_heads": 1}, vocab=["<s>", "a", "b", "c"], template=None), t


def the_file(path, t, tokens=("<s>", "a", "b", "c"), poke=None):
    q = D.q8_0(t["model.layers.0.mlp.down_proj.weight"])
    if poke is not None:
        q = q.copy()
        q[poke] ^= 1
    write_gguf(path, [("blk.0.attn_norm.weight", 0, (64,), t["model.layers.0.input_layernorm.weight"].tobytes()),
                      ("blk.0.ffn_down.weight", 8, (8, 64), q.tobytes())], tokens=list(tokens))


def test_roundf_sends_halves_away_from_zero():
    assert D.roundf(np.array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49])).tolist() == [1, 2, 3, -1, -3, 0]


def test_q8_0_block_layout():
    x = np.arange(-16, 16, dtype=np.float32) * (127 / 16)                  # largest magnitude 127: the scale is 1
    b = D.q8_0(x)
    assert b.size == 34
    assert np.frombuffer(b[:2].tobytes(), "<f2")[0] == 1.0
    assert np.frombuffer(b[2:].tobytes(), np.int8).tolist() == D.roundf(x).astype(int).tolist()


def test_a_file_written_from_the_known_tensors_is_exact(tmp_path):
    k, t = known()
    the_file(tmp_path / "m.gguf", t)
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert r["exact"] and r["identical"] == r["tensors"] == 2 and r["vocab"] == "same" and r["template"] == "absent in both"


def test_one_changed_byte_is_not_exact_and_is_named(tmp_path):
    k, t = known()
    the_file(tmp_path / "m.gguf", t, poke=40)
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and r["identical"] == 1 and r["differs"][0][0] == "blk.0.ffn_down.weight"


def test_other_weights_are_not_exact(tmp_path):
    k, _ = known(0)
    _, other = known(1)
    the_file(tmp_path / "m.gguf", other)
    assert not D.check(str(tmp_path / "m.gguf"), k, encoders=[])["exact"]


def test_a_changed_token_list_is_not_exact(tmp_path):
    k, t = known()
    the_file(tmp_path / "m.gguf", t, tokens=("<s>", "a", "B", "c"))
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and r["vocab"] == "differs" and r["identical"] == 2


@pytest.mark.skipif(not os.environ.get("LINEAGE_GGML"), reason="needs a build of the reference encoder (LINEAGE_GGML)")
def test_a_k_quant_file_names_the_build_that_wrote_it(tmp_path):
    encoders = D.builds()
    k, t = known()
    w = (0.05 * np.random.default_rng(2).standard_normal((4, 256))).astype(np.float32)
    t["model.layers.0.mlp.down_proj.weight"] = w
    q = encoders[0](12, w, 4, 256, 4 * 144)                                 # Q4_K: 144 bytes per 256 numbers
    write_gguf(tmp_path / "m.gguf", [("blk.0.attn_norm.weight", 0, (64,), t["model.layers.0.input_layernorm.weight"].tobytes()),
                                     ("blk.0.ffn_down.weight", 12, (4, 256), q.tobytes())], tokens=["<s>", "a", "b", "c"])
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=encoders)
    assert r["exact"] and encoders[0].name in r["builds"]
