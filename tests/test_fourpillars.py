"""The statement plumbing shared with SASH's verified inference demo: canonical JSON, directory hash, envelope."""
import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lineage_check import fourpillars as fpl


def test_canonical_json_follows_rfc_8785():
    assert fpl.canonical_json({"b": 1, "a": [1.0, 1e21, 1e-7, 0.1, -3.5, 100.0], "é": None, "t": True}) == \
        '{"a":[1,1e+21,1e-7,0.1,-3.5,100],"b":1,"t":true,"é":null}'.encode()


def test_dirhash_depends_on_content_and_names_only(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text("{}")
    (tmp_path / "a" / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "model.safetensors").write_bytes(b"weights")      # written in the other order
    (tmp_path / "b" / "config.json").write_text("{}")
    assert fpl.dirhash_sha256(tmp_path / "a") == fpl.dirhash_sha256(tmp_path / "b")
    (tmp_path / "b" / "model.safetensors").write_bytes(b"weightz")
    assert fpl.dirhash_sha256(tmp_path / "a") != fpl.dirhash_sha256(tmp_path / "b")


def test_envelope_signature_covers_the_payload():
    key = Ed25519PrivateKey.generate()
    env = fpl.sign_envelope({"_type": fpl.STATEMENT_TYPE, "predicate": {"x": 1}}, key)
    payload = base64.b64decode(env["payload"])
    key.public_key().verify(base64.b64decode(env["signatures"][0]["sig"]), fpl.pae(fpl.PAYLOAD_TYPE, payload))
