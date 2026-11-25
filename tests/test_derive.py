"""Exact identity: a GGUF file written from the known model's own tensors re-derives; anything else does not; and the
box loads an exact file back into exactly the known model's numbers."""
import os
import re
import struct

import numpy as np
import pytest

from lineage_check import derive as D
from lineage_check import gguf as G


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


VOCAB = ["<s>", "a", "b", "c"]
NORM, DOWN = "model.layers.0.input_layernorm.weight", "model.layers.0.mlp.down_proj.weight"


def known(seed=0, extra=False):
    rng = np.random.default_rng(seed)
    t = {NORM: (1 + 0.1 * rng.standard_normal(64)).astype(np.float32), DOWN: (0.05 * rng.standard_normal((8, 64))).astype(np.float32)}
    if extra:
        t["model.layers.0.mlp.up_proj.weight"] = (0.05 * rng.standard_normal((64, 8))).astype(np.float32)
    return D.Known(t.get, {"model_type": "llama", "num_attention_heads": 1}, vocab=list(VOCAB), template=None, names=list(t)), t


def the_file(path, t, tokens=VOCAB, poke=None, arch="llama", more=()):
    q = D.q8_0(t[DOWN])
    if poke is not None:
        q = q.copy()
        q[poke] ^= 1
    write_gguf(path, [("blk.0.attn_norm.weight", 0, (64,), t[NORM].tobytes()), ("blk.0.ffn_down.weight", 8, (8, 64), q.tobytes()), *more],
               tokens=list(tokens), arch=arch)


def test_roundf_sends_halves_away_from_zero():
    assert D.roundf(np.array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49])).tolist() == [1, 2, 3, -1, -3, 0]


def test_q8_0_block_layout():
    x = np.arange(-16, 16, dtype=np.float32) * (127 / 16)                  # largest magnitude 127: the scale is 1
    b = D.q8_0(x)
    assert b.size == 34
    assert np.frombuffer(b[:2].tobytes(), "<f2")[0] == 1.0
    assert np.frombuffer(b[2:].tobytes(), np.int8).tolist() == D.roundf(x).astype(int).tolist()


def test_a_file_written_from_the_known_tensors_is_exact_and_says_what_it_holds(tmp_path):
    k, t = known()
    the_file(tmp_path / "m.gguf", t)
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert r["exact"] and r["reason"] is None and r["identical"] == r["tensors"] == 2 and r["vocab"] == "same"
    assert r["formats"] == {"F32": 1, "Q8_0": 1} and r["bits_per_weight"] == round(8 * (64 * 4 + 16 * 34) / (64 + 512), 3) and len(r["sha256"]) == 64


def test_one_changed_byte_is_not_exact_and_is_named(tmp_path):
    k, t = known()
    the_file(tmp_path / "m.gguf", t, poke=40)
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and r["identical"] == 1 and r["differs"][0][0] == "blk.0.ffn_down.weight" and "sha256" not in r


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


def test_a_file_without_some_of_the_known_tensors_is_not_exact(tmp_path):
    k, t = known(extra=True)                                               # the known model has a third tensor the file lacks
    the_file(tmp_path / "m.gguf", t)
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and r["missing"] == ["model.layers.0.mlp.up_proj.weight"] and r["identical"] == 2


def test_a_file_with_a_tensor_the_known_model_lacks_is_not_exact(tmp_path):
    k, t = known()
    the_file(tmp_path / "m.gguf", t, more=[("blk.0.ffn_gate.weight", 0, (8, 8), np.zeros((8, 8), np.float32).tobytes())])
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and r["extra"] == ["model.layers.0.mlp.gate_proj.weight"]


def test_a_tensor_the_box_cannot_read_keeps_a_file_from_being_exact(tmp_path):
    k, t = known(extra=True)
    up = ("blk.0.ffn_up.weight", 21, (64, 256), bytes(64 * 110))           # IQ3_S: the encoder writes it, the box's reader does not
    the_file(tmp_path / "m.gguf", t, more=[up])
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and r["unservable"] and "cannot read" in r["reason"]


def test_a_relabelled_architecture_is_not_exact(tmp_path):
    """The review's attack: a Llama model's file labelled qwen2, so no row order would be undone on one side only."""
    k, t = known()
    the_file(tmp_path / "m.gguf", t, arch="qwen2")
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert not r["exact"] and "architecture" in r["reason"]


# ------------------------------------------------------------------ the check and the box's loader agree
INV_BLOCK = {v: k for k, v in G._BLOCK_NAMES.items()}
INV_TOP = {v: k for k, v in G._TOP_NAMES.items()}


def gguf_name(hub):
    if hub in INV_TOP:
        return INV_TOP[hub]
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)\.(weight|bias)", hub)
    return f"blk.{m.group(1)}.{INV_BLOCK[m.group(2)]}.{m.group(3)}"


def as_llama_cpp_writes(model, path, arch, interleave):
    """The model's weights in a GGUF file as llama.cpp's converter writes them (float32), query and key rows interleaved
    for the Llama family."""
    cfg = model.config.to_dict()
    tensors = []
    for hub, p in model.state_dict().items():
        w = p.detach().numpy().astype(np.float32)
        h = D.heads_of(cfg, hub) if interleave else None
        if h:
            w = w.reshape(h, 2, w.shape[0] // h // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
        tensors.append((gguf_name(hub), 0, w.shape, np.ascontiguousarray(w).tobytes()))
    vocab = [f"t{i}" for i in range(cfg["vocab_size"])]
    write_gguf(path, tensors, tokens=vocab, arch=arch)
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    return D.Known(sd.get, cfg, vocab=vocab, names=list(sd))


@pytest.mark.parametrize("kind", ["llama", "mistral", "qwen2"])
def test_an_exact_file_loads_back_into_exactly_the_known_model(tmp_path, kind):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from lineage_check import outputs as O
    from lineage_check import remote as R
    size = dict(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=64, tie_word_embeddings=False)
    config = {"llama": transformers.LlamaConfig, "mistral": transformers.MistralConfig, "qwen2": transformers.Qwen2Config}[kind](**size)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    arch = D.GGUF_ARCH[kind]
    k = as_llama_cpp_writes(model, tmp_path / "m.gguf", arch, D.interleaved(kind))
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=[])
    assert r["exact"], r["reason"]
    served = O.load_like(model, R.LocalGGUF([str(tmp_path / "m.gguf")]))
    for name, p in model.state_dict().items():
        assert torch.equal(served.state_dict()[name], p), name                # the box serves the known model's own numbers


@pytest.mark.skipif(not os.environ.get("LINEAGE_GGML"), reason="needs a build of the reference encoder (LINEAGE_GGML)")
def test_a_k_quant_file_names_the_build_that_wrote_it(tmp_path):
    encoders = D.builds()
    k, t = known()
    w = (0.05 * np.random.default_rng(2).standard_normal((4, 256))).astype(np.float32)
    t[DOWN] = w
    q = encoders[0](12, w, 4, 256, 4 * 144)                                 # Q4_K: 144 bytes per 256 numbers
    write_gguf(tmp_path / "m.gguf", [("blk.0.attn_norm.weight", 0, (64,), t[NORM].tobytes()),
                                     ("blk.0.ffn_down.weight", 12, (4, 256), q.tobytes())], tokens=list(VOCAB))
    r = D.check(str(tmp_path / "m.gguf"), k, encoders=encoders)
    assert r["exact"] and encoders[0].name in r["builds"]
