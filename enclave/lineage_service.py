"""lineage-check as an enclave service, in the shape of SASH's verified inference demo.

The model under audit is mounted read-only (in Tinfoil: a Modelwrap pack at /tinfoil/mpk/mpk-<root hash>) and
hashed at boot under dirhash-sha256/1, as the inference demo hashes its base. A verifier POSTs a fresh nonce; the
service derives the secret tokens and layers from it, fingerprints the model against every listed model, and
returns one signed statement: an in-toto Statement v1 in a DSSE envelope, signed with the Tinfoil attested key,
whose `scannedModel` uses the inference statement's component shape. The statement says match / no match, which
entry, and whether the refusal-ablation signature is present. No scores, tokens or weights leave the enclave.

  GET  /health
  GET  /v1/model       the scanned model's identity, computed at boot
  POST /v1/challenge   {"nonce": "<32-128 hex>"}  ->  {"envelope": <DSSE>, "publicKey": "<hex>"}

Settings (pinned in tinfoil-config.yml, so measured):
  MODEL_DIR, MODEL_NAME, MODEL_REF (repo@revision), MODEL_SCHEMA (modelwrap pack schema)
  CONCERN_LIST (path), CONCERN_LIST_DIGEST (jcs-sha256/1 of it; the service refuses to start on a mismatch)
  TOKENS (512), LAYERS (4), PORT (8080), ALLOW_DEV_KEY (1 outside an enclave: statements say keyBinding null)
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from lineage_check import fingerprint as fp4
from lineage_check import fourpillars as FP
from lineage_check import protocol as LC
from lineage_check import remote as R

PREDICATE_TYPE = "urn:lineage-check:LineageStatement:v0.2"
NONCE = re.compile(r"[0-9a-f]{32,128}")
DEBUG = os.environ.get("DEBUG") == "1"  # local twin only: per-entry distances to stderr (pinned off in the enclave)
torch.set_grad_enabled(False)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source(entry_or_dir, label=None):
    """A listed model: a mounted pack (`path`) in the enclave, or a pinned Hub revision in the local twin."""
    if isinstance(entry_or_dir, (str, Path)):
        return R.LocalST(str(entry_or_dir), label=label)
    if entry_or_dir.get("path"):
        return R.LocalST(entry_or_dir["path"], label=entry_or_dir["id"])
    if not entry_or_dir.get("revision"):
        raise ValueError(f"listed model {entry_or_dir['id']} has no pinned revision")
    return R.Remote(entry_or_dir["repo"], entry_or_dir["revision"])


def scan(q, listed, nonce, n, k):
    """lineage_check.scan_one on a mounted model: (verdict, matched id, ablation signature, layers read)."""
    Lq = q.config["num_hidden_layers"]
    q_layers = LC.layer_plan(Lq, nonce, k)
    verdict, matched, ablated, fq_cache = "no_match", None, None, {}
    for e, ref in listed:
        strings = fp4.select_strings(ref.vocab, nonce, n)
        key = tuple(strings)
        if key not in fq_cache:
            fq_cache[key] = LC.fingerprint(q, strings, q_layers, forensic=ablated is None)
            ablated = fq_cache[key][0].get("ablation_signature", ablated)
        fq, kq = fq_cache[key]
        fr, kr = LC.fingerprint(ref, strings, LC.anchor_layers(ref.config["num_hidden_layers"], Lq, q_layers))
        common = sorted(set(kq) & set(kr))
        if len(common) < 0.5 * len(strings):
            continue                                   # a different tokenizer: not comparable on these tokens
        r = fp4.compare(LC.restrict(fr, kr, common), LC.restrict(fq, kq, common), align=True)
        d = float(np.mean([r["qk"], r["ov"]]))
        if DEBUG:
            print(f"  {e['id']}: distance {d:.4f} (threshold {e['threshold']}), {len(common)} tokens", file=sys.stderr, flush=True)
        if d < e["threshold"]:
            verdict, matched = "match", e["id"]
            break
    return verdict, matched, bool(ablated), q_layers


class Service:
    def __init__(self, env=os.environ):
        self.model_dir = Path(env["MODEL_DIR"])
        self.model_name = env.get("MODEL_NAME") or self.model_dir.name
        self.model_ref = env.get("MODEL_REF") or None
        self.schema = env.get("MODEL_SCHEMA", "2")
        self.n, self.k = int(env.get("TOKENS", 512)), int(env.get("LAYERS", 4))
        self.key, self.attested = FP.load_signing_key(Path(env.get("SIGNING_KEY_PATH", FP.ATTESTED_KEY_PATH)),
                                                      env.get("ALLOW_DEV_KEY") == "1")
        self.concern = json.load(open(env["CONCERN_LIST"]))
        self.concern_digest = FP.jcs_sha256(self.concern)
        pinned = env.get("CONCERN_LIST_DIGEST")
        if pinned and pinned != self.concern_digest:
            raise RuntimeError(f"concern list digest {self.concern_digest} does not match the pinned {pinned}")
        self.started_at = now()
        t0 = time.time()
        self.dirhash = FP.dirhash_sha256(self.model_dir)   # read-only mount: the scan reads exactly these bytes
        self.hash_seconds = round(time.time() - t0, 1)
        self.q = source(self.model_dir, self.model_name)
        self.listed = [(e, source(e)) for e in self.concern["models"]]
        self.used, self.lock = set(), threading.Lock()

    def component(self):
        """The scanned model, in the inference statement's component shape (so `identifiers()` links them)."""
        aliases = []
        if self.model_ref:
            aliases.append({"scheme": FP.HF_REVISION, "ref": self.model_ref})
        if self.model_dir.name.startswith(FP.MPK_MOUNT_PREFIX):
            aliases.append({"scheme": f"modelwrap/{self.schema}", "ref": self.model_dir.name.removeprefix(FP.MPK_MOUNT_PREFIX)})
        return {"id": "scanned", "role": "base", "name": self.model_name, "scheme": FP.DIRHASH,
                "digest": self.dirhash, "alsoKnownAs": aliases}

    def challenge(self, nonce):
        if not NONCE.fullmatch(nonce or ""):
            raise ValueError("the nonce must be 32 to 128 lowercase hex characters")
        with self.lock:
            if nonce in self.used:
                raise ValueError("nonce already used")
            self.used.add(nonce)
            t0 = time.time()
            verdict, matched, ablated, layers = scan(self.q, self.listed, nonce, self.n, self.k)
            seconds = round(time.time() - t0, 1)
        pub = FP.raw_public_key(self.key)
        binding = FP.fetch_key_binding(pub, FP.jcs_sha256({"lineageChallenge": nonce})) if self.attested else None
        model = self.component()
        statement = {
            "_type": FP.STATEMENT_TYPE,
            "subject": [{"name": model["name"], "digest": {"sha256": model["digest"]}}],
            "predicateType": PREDICATE_TYPE,
            "predicate": {
                "scannedModel": model,
                "concernList": {"name": self.concern["name"], "scheme": FP.JCS, "digest": self.concern_digest,
                                "entries": len(self.concern["models"])},
                "challenge": {"nonce": nonce},
                "lineage": {"verdict": verdict, "matched": matched, "refusalAblationSignature": ablated},
                "method": {"name": "token-anchored circuits", "version": fp4.VERSION, "tokens": self.n,
                           "layersSampled": len(layers), "layerRule": LC.LAYER_RULE},
                "execution": {"platform": "tinfoil-containers" if binding else "local", "startedAt": self.started_at,
                              "issuedAt": now(), "scanSeconds": seconds, "statementKey": pub.hex(),
                              "attestation": {"keyBinding": binding}},
            },
        }
        return {"envelope": FP.sign_envelope(statement, self.key), "publicKey": pub.hex()}


def serve(svc, port):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if self.path == "/v1/model":
                return self._send(200, {"model": svc.component(), "hashSeconds": svc.hash_seconds,
                                        "concernList": {"digest": svc.concern_digest}})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/challenge":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self._send(200, svc.challenge(body.get("nonce")))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:  # a scan failure is an error, never a verdict
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (now(), fmt % args))

    print(f"{now()} ready: {svc.model_name} dirhash {svc.dirhash[:16]}... ({svc.hash_seconds}s), "
          f"{len(svc.listed)} listed models, concern list {svc.concern_digest[:16]}...", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    serve(Service(), int(os.environ.get("PORT", 8080)))
