"""Statements: signed, tamper-evident, bound to one nonce. The audit draw is fixed by the nonce."""
import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lineage_check import fingerprint, protocol

STATEMENT = {"_type": "https://in-toto.io/Statement/v1", "subject": [{"name": "org/model", "digest": {"gitCommit": "abc"}}],
             "predicateType": protocol.PREDICATE, "predicate": {"verdict": "match", "matched": "x", "nonce": "aa" * 16}}


def test_signed_statement_opens_under_its_key():
    sk = Ed25519PrivateKey.generate()
    assert protocol.open_envelope(protocol.sign(STATEMENT, sk), sk.public_key(), "aa" * 16) == STATEMENT


def test_tampered_statement_is_rejected():
    sk = Ed25519PrivateKey.generate()
    env = protocol.sign(STATEMENT, sk)
    forged = json.loads(base64.b64decode(env["payload"]))
    forged["predicate"]["verdict"] = "no_match"
    env["payload"] = base64.b64encode(json.dumps(forged, sort_keys=True).encode()).decode()
    with pytest.raises(ValueError, match="signature"):
        protocol.open_envelope(env, sk.public_key())


def test_wrong_key_is_rejected():
    env = protocol.sign(STATEMENT, Ed25519PrivateKey.generate())
    with pytest.raises(ValueError, match="signature"):
        protocol.open_envelope(env, Ed25519PrivateKey.generate().public_key())


def test_replayed_statement_is_rejected():
    sk = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="replay"):
        protocol.open_envelope(protocol.sign(STATEMENT, sk), sk.public_key(), "bb" * 16)


def test_layers_are_fixed_by_the_nonce():
    assert protocol.layer_plan(32, "n1", 4) == protocol.layer_plan(32, "n1", 4)
    assert len({tuple(protocol.layer_plan(32, f"n{i}", 4)) for i in range(20)}) > 10
    assert protocol.layer_plan(3, "n1", 4) == [0, 1, 2]


def test_tokens_are_fixed_by_the_nonce():
    vocab = {f"Ġword{chr(97 + i % 26)}{chr(97 + i // 26)}": 300 + i for i in range(600)}
    a = fingerprint.select_strings(vocab, "n1", 64)
    assert a == fingerprint.select_strings(vocab, "n1", 64) and a != fingerprint.select_strings(vocab, "n2", 64)
