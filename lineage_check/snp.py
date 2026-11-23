"""AMD SEV-SNP attestation reports: read one, and check it against AMD's certificate chain.

A report is 1,184 bytes signed by a key that belongs to one chip at one firmware version (ECDSA P-384). Two of its
fields carry what a verifier of a lineage statement needs:

  report_data   64 bytes the guest chooses when it asks for the report. The service puts the SHA-256 of
                {statement key, challenge nonce} there, so the report names the key that signed the statement.
  host_data     32 bytes the host fixes when it launches the guest. On Azure confidential containers this is the
                SHA-256 of the policy that pins every container image, command and environment variable.

`verify` checks the signatures (report <- chip key <- AMD intermediate <- AMD root, the root pinned below) and
returns the fields. The launch measurement is the base system the containers run on. On Azure that is Microsoft's
utility VM, and `verify_uvm_endorsement` checks Microsoft's signed statement that it built that measurement.
"""
from __future__ import annotations

import base64
import binascii
import datetime
import hashlib
import json
import os
import struct
import urllib.request

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, utils

from . import fourpillars as FP

REPORT_SIZE, SIGNED = 0x4A0, 0x2A0
PLATFORM_ACI = "azure-aci-sev-snp"
# SHA-256 of AMD's root keys (SubjectPublicKeyInfo, DER), computed from
# https://kdsintf.amd.com/vcek/v1/<product>/cert_chain on 2026-11-23.
AMD_ROOTS = {
    "9f056bee44377e29308cb5ffa895bdfb62d18881fa6bed8d6f075b0204089cb9": "ARK-Milan",
    "429a69c9422aa258ee4d8db5fcda9c6470ef15f8cd5a9cebd6cbc7d90b863831": "ARK-Genoa",
    "4f125410563a2ab9a50356f9243f6fe0b6f73de98603f53f90339c70e9d7ad08": "ARK-Turin",
}


# SHA-256 of the certificate "Microsoft Supply Chain RSA Root CA 2022" (DER), computed from
# https://www.microsoft.com/pkiops/certs/ on 2026-11-23. Azure's endorsements name the same value as their issuer.
MICROSOFT_UVM_ROOT = "23ffe2b8bdb9a1711515d4cffda04bc7f793d513c76c243f1020507d8669b7db"
UVM_FEED, UVM_EKU = "ContainerPlat-AMD-UVM", "1.3.6.1.4.1.311.76.59.1.2"
_COSE_HASH = {-37: hashes.SHA256, -38: hashes.SHA384, -39: hashes.SHA512}    # RSASSA-PSS (RFC 8230)


class AttestationError(Exception):
    pass


def parse(raw: bytes) -> dict:
    """The fields of an attestation report (AMD SEV-SNP ABI, ATTESTATION_REPORT). No signature check."""
    if len(raw) != REPORT_SIZE:
        raise AttestationError(f"a report is {REPORT_SIZE} bytes, got {len(raw)}")
    version, guest_svn, policy = struct.unpack_from("<IIQ", raw, 0)
    vmpl, algorithm = struct.unpack_from("<II", raw, 0x30)
    return {"version": version, "guest_svn": guest_svn, "policy": policy, "debug": bool(policy >> 19 & 1),
            "vmpl": vmpl, "signature_algorithm": algorithm,
            "report_data": raw[0x50:0x90], "measurement": raw[0x90:0xC0], "host_data": raw[0xC0:0xE0],
            "id_key_digest": raw[0xE0:0x110], "author_key_digest": raw[0x110:0x140], "report_id": raw[0x140:0x160],
            "reported_tcb": raw[0x180:0x188], "chip_id": raw[0x1A0:0x1E0]}


def _signed_by(child: x509.Certificate, parent: x509.Certificate) -> None:
    parent.public_key().verify(child.signature, child.tbs_certificate_bytes,
                               padding.PSS(mgf=padding.MGF1(hashes.SHA384()), salt_length=48), hashes.SHA384())


def certificates(blob: bytes) -> list[x509.Certificate]:
    """Chip key, intermediate, root: from a PEM bundle, or from Azure's JSON wrapping of one."""
    try:
        doc = json.loads(blob)
        blob = (doc["vcekCert"] + doc["certificateChain"]).encode()
    except (ValueError, KeyError, TypeError):
        pass
    return x509.load_pem_x509_certificates(blob)


def verify(raw: bytes, chain: bytes, roots: dict | None = None) -> dict:
    """Check the report's signature chain down to a pinned AMD root, and return the report's fields."""
    certs = certificates(chain)
    if len(certs) < 3:
        raise AttestationError("the chain needs the chip key, AMD's intermediate and AMD's root")
    leaf, mid, root = certs[:3]
    spki = root.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    name = (AMD_ROOTS if roots is None else roots).get(hashlib.sha256(spki).hexdigest())
    if name is None:
        raise AttestationError("the chain does not end in a pinned AMD root")
    fields = parse(raw)
    if fields["signature_algorithm"] != 1:
        raise AttestationError("the report is not signed with ECDSA P-384")
    r = int.from_bytes(raw[SIGNED:SIGNED + 72], "little")
    s = int.from_bytes(raw[SIGNED + 72:SIGNED + 144], "little")
    try:
        _signed_by(root, root)
        _signed_by(mid, root)
        _signed_by(leaf, mid)
        leaf.public_key().verify(utils.encode_dss_signature(r, s), raw[:SIGNED], ec.ECDSA(hashes.SHA384()))
    except InvalidSignature:
        raise AttestationError("a signature in the chain does not verify") from None
    if fields["debug"]:
        raise AttestationError("the guest was launched with debugging allowed")
    return {**fields, "root": name}


def binding_payload(statement_key: bytes, nonce: str) -> bytes:
    """What a report's report_data commits to: this statement key, for this challenge."""
    return FP.canonical_json({"lineageChallenge": nonce, "statementKey": statement_key.hex()})


def _decode(text: str) -> bytes:
    """Base64 (either alphabet). A report may also come as hex, which is how the sidecar's raw call returns it."""
    if len(text) == 2 * REPORT_SIZE:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_" if ("-" in text or "_" in text) else None)
    except binascii.Error:
        raise AttestationError("a field of the attestation document is not base64") from None


def _cbor(buf: bytes, i: int = 0):
    """Decode one CBOR item: the subset a COSE_Sign1 document uses. Returns (value, next index)."""
    major, info = buf[i] >> 5, buf[i] & 31
    i += 1
    if info < 24:
        n = info
    elif info <= 27:
        size = 1 << (info - 24)
        n, i = int.from_bytes(buf[i:i + size], "big"), i + size
    else:
        raise AttestationError("unsupported CBOR length")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major == 2:
        return bytes(buf[i:i + n]), i + n
    if major == 3:
        return bytes(buf[i:i + n]).decode(), i + n
    if major == 4:
        out = []
        for _ in range(n):
            v, i = _cbor(buf, i)
            out.append(v)
        return out, i
    if major == 5:
        out = {}
        for _ in range(n):
            k, i = _cbor(buf, i)
            out[k], i = _cbor(buf, i)
        return out, i
    if major == 6:
        return _cbor(buf, i)                                      # a tag: keep the tagged value
    return {20: False, 21: True, 22: None}.get(info), i


def _head(major: int, n: int) -> bytes:
    if n < 24:
        return bytes([major << 5 | n])
    size = 1 if n < 1 << 8 else 2 if n < 1 << 16 else 4 if n < 1 << 32 else 8
    return bytes([major << 5 | {1: 24, 2: 25, 4: 26, 8: 27}[size]]) + n.to_bytes(size, "big")


def verify_uvm_endorsement(blob: bytes, measurement: bytes, root_sha256: str = MICROSOFT_UVM_ROOT) -> dict:
    """Check Microsoft's signed statement (COSE_Sign1, RFC 9052) that `measurement` is a utility VM it built."""
    try:
        (protected_raw, _, payload, signature), _ = _cbor(blob)
        protected, _ = _cbor(protected_raw)
        chain = protected[33] if isinstance(protected[33], list) else [protected[33]]
        chain = [x509.load_der_x509_certificate(c) for c in chain]
        algorithm = _COSE_HASH[protected[1]]()
    except (ValueError, KeyError, TypeError, IndexError):
        raise AttestationError("the endorsement is not a COSE_Sign1 document with a certificate chain") from None
    leaf, root = chain[0], chain[-1]
    if hashlib.sha256(root.public_bytes(serialization.Encoding.DER)).hexdigest() != root_sha256:
        raise AttestationError("the endorsement's chain does not end in the pinned Microsoft root")
    when = datetime.datetime.fromtimestamp(protected.get("signingtime", 0), datetime.timezone.utc)
    to_sign = (b"\x84" + _head(3, 10) + b"Signature1" + _head(2, len(protected_raw)) + protected_raw + _head(2, 0)
               + _head(2, len(payload)) + payload)
    try:
        for child, parent in zip(chain, chain[1:] + [root]):
            child.verify_directly_issued_by(parent)
            if protected.get("signingtime") and not child.not_valid_before_utc <= when <= child.not_valid_after_utc:
                raise AttestationError("a certificate in the endorsement's chain was not valid when it was signed")
        usage = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        leaf.public_key().verify(signature, to_sign, padding.PSS(mgf=padding.MGF1(algorithm), salt_length=algorithm.digest_size),
                                 algorithm)
    except (InvalidSignature, ValueError, TypeError, x509.ExtensionNotFound):
        raise AttestationError("the endorsement's signature or chain does not verify") from None
    if x509.ObjectIdentifier(UVM_EKU) not in usage or protected.get("feed") != UVM_FEED:
        raise AttestationError("the endorsement is not for Azure's confidential container utility VM")
    body = json.loads(payload)
    if body.get("x-ms-sevsnpvm-launchmeasurement") != measurement.hex():
        raise AttestationError("the endorsement is for a different launch measurement")
    return {"feed": protected["feed"], "guest_svn": body.get("x-ms-sevsnpvm-guestsvn"), "issuer": protected.get("iss"),
            "signed": when.strftime("%Y-%m-%d") if protected.get("signingtime") else None}


def aci_attest(payload: bytes, url: str | None = None, timeout: float = 30) -> dict:
    """A fresh report over `payload`, from the attestation sidecar of an Azure confidential container group."""
    url = (url or os.environ.get("ATTEST_URL", "http://localhost:8080")).rstrip("/")
    request = urllib.request.Request(url + "/attest/combined", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"runtime_data": base64.b64encode(payload).decode()}).encode())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return {"platform": PLATFORM_ACI, "runtimeData": base64.b64encode(payload).decode(), "report": body["evidence"],
            "certChain": body["endorsements"], "uvmEndorsements": body.get("uvm_endorsements"),
            "endorsedTcb": body.get("endorsed_tcb")}


def verify_binding(document: dict, statement_key: bytes, nonce: str, policy: bytes | None = None,
                   roots: dict | None = None) -> dict:
    """Check that a hardware report names `statement_key` for `nonce` (and, if given, was launched under `policy`)."""
    payload = _decode(document["runtimeData"])
    if payload != binding_payload(statement_key, nonce):
        raise AttestationError("the report was not requested for this statement key and this challenge")
    fields = verify(_decode(document["report"]), _decode(document["certChain"]), roots)
    if fields["report_data"] != hashlib.sha256(payload).digest() + bytes(32):
        raise AttestationError("the report's report_data is not the hash of the statement key and the challenge")
    if policy is not None and fields["host_data"] != hashlib.sha256(policy).digest():
        raise AttestationError("the report's host_data is not the hash of the expected policy")
    if document.get("uvmEndorsements"):
        fields["utility_vm"] = verify_uvm_endorsement(_decode(document["uvmEndorsements"]), fields["measurement"])
    return fields
