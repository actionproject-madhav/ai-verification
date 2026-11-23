"""AMD SEV-SNP attestation reports: read one, and check it against AMD's certificate chain.

A report is 1,184 bytes signed by a key that belongs to one chip at one firmware version (ECDSA P-384). Two of its
fields carry what a verifier of a lineage statement needs:

  report_data   64 bytes the guest chooses when it asks for the report. The service puts the SHA-256 of
                {statement key, challenge nonce} there, so the report names the key that signed the statement.
  host_data     32 bytes the host fixes when it launches the guest. On Azure confidential containers this is the
                SHA-256 of the policy that pins every container image, command and environment variable.

`verify` checks the signatures (report <- chip key <- AMD intermediate <- AMD root, the root pinned below) and
returns the fields. It does not judge the launch measurement: on Azure that is Microsoft's utility VM, which
Microsoft endorses in a separate signed document.
"""
from __future__ import annotations

import base64
import binascii
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
    try:
        return bytes.fromhex(text)
    except ValueError:
        pass
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_" if ("-" in text or "_" in text) else None)
    except binascii.Error:
        raise AttestationError("a field of the attestation document is neither hex nor base64") from None


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
    return fields
