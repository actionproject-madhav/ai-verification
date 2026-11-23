"""Four Pillars statement plumbing, copied from SASH's verified inference demo so lineage statements are
byte-compatible with its inference statements.

Source: github.com/SG-AI-Safety-Hub/inference-demo, service/src/inference_service/{identity,statement,settings}.py
License: Apache License 2.0 (inference-demo/LICENSE). Copyright the SG AI Safety Hub authors.
Changes: one file instead of three; load_signing_key takes a path and a flag instead of Settings;
fetch_key_binding takes the attestation nonce (so a report can be bound to a verifier's challenge).
verify_statement.py checks this copy against the upstream modules.
"""

import base64
import binascii
import hashlib
import http.client
import json
import math
import secrets
import socket
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

# ---------------------------------------------------------------- identity.py
DIRHASH = "dirhash-sha256/1"
JCS = "jcs-sha256/1"


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dirhash_sha256(root: Path, files: Iterable[Path] | None = None) -> str:
    """dirhash-sha256/1: sha256 of the canonical JSON of {relative path: file sha256}."""
    if files is None:
        files = (p for p in root.rglob("*") if p.is_file())
    digests = {str(p.relative_to(root)): file_sha256(p) for p in sorted(files)}
    return hashlib.sha256(canonical_json(digests)).hexdigest()


def jcs_sha256(value) -> str:
    """jcs-sha256/1: sha256 of the canonical JSON of `value`."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonical_json(value) -> bytes:
    """RFC 8785 (JCS)."""
    return _jcs(value).encode()


def _jcs(value) -> str:
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: item[0].encode("utf-16-be"))
        return "{" + ",".join(f"{_jcs(k)}:{_jcs(v)}" for k, v in items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_jcs(v) for v in value) + "]"
    if isinstance(value, float):
        return _jcs_number(value)
    return json.dumps(value, ensure_ascii=False)


def _jcs_number(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"JSON has no {x}")
    if x == 0:
        return "0"
    sign, digit_tuple, exponent = Decimal(repr(x)).as_tuple()
    n = len(digit_tuple) + exponent
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        text = digits + "0" * (n - k)
    elif 0 < n <= 21:
        text = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        text = "0." + "0" * -n + digits
    else:
        mantissa = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
        text = f"{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"
    return ("-" if sign else "") + text


# ---------------------------------------------------------------- settings.py
ATTESTED_KEY_ID = "enclave-signing-key"
ATTESTED_KEY_PATH = Path("/run/tinfoil/keys") / ATTESTED_KEY_ID / "private_key.pem"

# ---------------------------------------------------------------- statement.py
STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"
HF_REVISION = "hf-revision/1"
MPK_MOUNT_PREFIX = "mpk-"
ATTESTATION_SOCKET = Path("/tinfoil/attestation.sock")
REPORT_DATA_V1 = "https://tinfoil.sh/report-data/v1"
CRYPTO_MATERIAL_V1 = "https://tinfoil.sh/crypto-material/v1"
KEY_SPKI_V1 = "https://tinfoil.sh/key/spki/v1"


class KeyBindingError(Exception):
    """Tinfoil's report doesn't list this enclave's signing key."""


def load_signing_key(path: Path = ATTESTED_KEY_PATH, allow_dev_key: bool = False) -> tuple[Ed25519PrivateKey, bool]:
    """The key that signs statements, and whether it is Tinfoil's attested key."""
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except FileNotFoundError:
        if not allow_dev_key:
            raise RuntimeError(f"no attested key at {path}: the config must declare an ed25519 attested key "
                               f"{ATTESTED_KEY_ID!r} and grant it to this container")
        return Ed25519PrivateKey.generate(), False
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError(f"the key at {path} is not an ed25519 key")
    return key, True


def raw_public_key(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes_raw()


def fetch_key_binding(public_key: bytes, nonce_hex: str | None = None, socket_path: Path = ATTESTATION_SOCKET) -> dict:
    """A fresh v3 attestation document that lists `public_key` as our attested key."""
    connection = _UnixHTTPConnection(socket_path)
    try:
        connection.request("GET", f"/.well-known/tinfoil-attestation?nonce={nonce_hex or secrets.token_hex(32)}")
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise KeyBindingError(f"local attestation returned HTTP {response.status}")
    document = json.loads(body)
    if bound_key(document) != public_key:
        raise KeyBindingError("Tinfoil's report does not list this enclave's signing key")
    return document


def bound_key(document: dict) -> bytes:
    """The raw Ed25519 key a v3 document lists under our key id, once its report data checks out."""
    challenge = document.get("challenge") or {}
    if challenge.get("report_data_algorithm") != REPORT_DATA_V1:
        raise KeyBindingError(f"unknown report data algorithm {challenge.get('report_data_algorithm')!r}")
    try:
        digest = hashlib.sha256(REPORT_DATA_V1.encode())
        digest.update(bytes.fromhex(challenge["nonce"]))
        for section in ("crypto_material", "device_evidence"):
            digest.update(hashlib.sha256(base64.b64decode(document[section])).digest())
        material = json.loads(base64.b64decode(document["crypto_material"]))
    except (KeyError, ValueError, binascii.Error) as e:
        raise KeyBindingError(f"malformed attestation document: {e}") from e
    if challenge.get("report_data") != (digest.digest() + bytes(32)).hex():
        raise KeyBindingError("the report data is not derived from the nonce and sections")
    if material.get("format") != CRYPTO_MATERIAL_V1:
        raise KeyBindingError(f"unknown crypto_material {material.get('format')!r}")
    for item in material.get("items") or []:
        if item.get("id") == ATTESTED_KEY_ID and item.get("format") == KEY_SPKI_V1:
            try:
                key = serialization.load_der_public_key(bytes.fromhex(item.get("data", "")))
            except ValueError as e:
                raise KeyBindingError(f"malformed attested key: {e}") from e
            if not isinstance(key, Ed25519PublicKey):
                raise KeyBindingError("the attested key is not an ed25519 key")
            return key.public_bytes_raw()
    raise KeyBindingError(f"the report lists no attested key {ATTESTED_KEY_ID!r}")


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path):
        super().__init__("localhost", timeout=30)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding."""
    return b"DSSEv1 %d %s %d %s" % (len(payload_type), payload_type.encode(), len(payload), payload)


def sign_envelope(statement: dict, key: Ed25519PrivateKey) -> dict:
    payload = canonical_json(statement)
    signature = key.sign(pae(PAYLOAD_TYPE, payload))
    return {"payloadType": PAYLOAD_TYPE, "payload": base64.b64encode(payload).decode(),
            "signatures": [{"sig": base64.b64encode(signature).decode()}]}
