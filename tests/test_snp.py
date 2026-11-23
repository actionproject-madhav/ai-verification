"""Hardware reports: a report made here, signed by a chain made here, must pass; any tampering must fail."""
import base64
import datetime
import hashlib

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from lineage_check import snp

PSS = padding.PSS(mgf=padding.MGF1(hashes.SHA384()), salt_length=48)
NONCE = "ab" * 16


def _cert(name, key, issuer_name, issuer_key):
    now = datetime.datetime(2026, 1, 1)
    return (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)]))
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=3650))
            .sign(issuer_key, hashes.SHA384(), rsa_padding=PSS))


@pytest.fixture(scope="module")
def chain():
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    mid_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    chip_key = ec.generate_private_key(ec.SECP384R1())
    certs = [_cert("chip", chip_key, "mid", mid_key), _cert("mid", mid_key, "root", root_key),
             _cert("root", root_key, "root", root_key)]
    pem = b"".join(c.public_bytes(serialization.Encoding.PEM) for c in certs)
    spki = certs[2].public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return chip_key, pem, {hashlib.sha256(spki).hexdigest(): "test root"}


def report(chip_key, report_data=bytes(64), host_data=bytes(32), policy=0x30000):
    raw = bytearray(snp.REPORT_SIZE)
    raw[0:4] = (2).to_bytes(4, "little")
    raw[8:16] = policy.to_bytes(8, "little")
    raw[0x34:0x38] = (1).to_bytes(4, "little")
    raw[0x50:0x90] = report_data
    raw[0xC0:0xE0] = host_data
    r, s = utils.decode_dss_signature(chip_key.sign(bytes(raw[:snp.SIGNED]), ec.ECDSA(hashes.SHA384())))
    raw[snp.SIGNED:snp.SIGNED + 72] = r.to_bytes(72, "little")
    raw[snp.SIGNED + 72:snp.SIGNED + 144] = s.to_bytes(72, "little")
    return bytes(raw)


def document(chip_key, pem, key, nonce=NONCE, policy=b"policy"):
    payload = snp.binding_payload(key, nonce)
    raw = report(chip_key, hashlib.sha256(payload).digest() + bytes(32), hashlib.sha256(policy).digest())
    return {"platform": snp.PLATFORM_ACI, "runtimeData": base64.b64encode(payload).decode(),
            "report": base64.b64encode(raw).decode(), "certChain": base64.b64encode(pem).decode()}


def test_a_report_signed_by_the_chain_verifies(chain):
    chip_key, pem, roots = chain
    fields = snp.verify(report(chip_key, host_data=b"h" * 32), pem, roots)
    assert fields["host_data"] == b"h" * 32 and fields["root"] == "test root" and not fields["debug"]


def test_a_changed_report_fails(chain):
    chip_key, pem, roots = chain
    raw = bytearray(report(chip_key))
    raw[0xC0] ^= 1
    with pytest.raises(snp.AttestationError):
        snp.verify(bytes(raw), pem, roots)


def test_a_chain_from_an_unpinned_root_fails(chain):
    chip_key, pem, _ = chain
    with pytest.raises(snp.AttestationError):
        snp.verify(report(chip_key), pem)                         # the real AMD roots do not include the test root


def test_a_debuggable_guest_fails(chain):
    chip_key, pem, roots = chain
    with pytest.raises(snp.AttestationError):
        snp.verify(report(chip_key, policy=0x30000 | 1 << 19), pem, roots)


def test_binding_names_the_key_the_challenge_and_the_policy(chain):
    chip_key, pem, roots = chain
    key = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    doc = document(chip_key, pem, key)
    assert snp.verify_binding(doc, key, NONCE, b"policy", roots)["root"] == "test root"
    other = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    for bad in (dict(statement_key=other, nonce=NONCE), dict(statement_key=key, nonce="cd" * 16),
                dict(statement_key=key, nonce=NONCE, policy=b"another policy")):
        with pytest.raises(snp.AttestationError):
            snp.verify_binding(doc, roots=roots, **bad)


def test_azure_wraps_the_chain_in_json(chain):
    chip_key, pem, roots = chain
    import json
    first, rest = pem.split(b"-----END CERTIFICATE-----\n", 1)
    wrapped = json.dumps({"vcekCert": (first + b"-----END CERTIFICATE-----\n").decode(), "tcbm": "0", "certificateChain": rest.decode()})
    assert snp.verify(report(chip_key), wrapped.encode(), roots)["root"] == "test root"
