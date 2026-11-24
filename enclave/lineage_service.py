"""lineage-check as an enclave service, in the shape of SASH's verified inference demo.

Two answers. MODE=match (the default) says whether the mounted model derives from an entry of one list.
MODE=light gives the traffic light: Red, Green or Amber against a watchlist and a registry of known models, with
the design-free fingerprint. In that mode the model is read by byte range at a pinned revision, so it can be far
larger than the enclave (the government-named models are 0.75 to 2.8 trillion parameters). With MODEL_DIR it is a
private model mounted in the enclave instead, named by the hash of its files.

Two platforms. In Tinfoil (the default) the model is a mounted pack and the platform supplies the attested key.
In an Azure confidential container group (PLATFORM=azure-aci) the service fetches the pinned revision itself,
makes its signing key inside the enclave, and binds it to every statement with a fresh AMD SEV-SNP report from
the attestation sidecar (ATTEST_URL).

The model under audit is mounted read-only (in Tinfoil: a Modelwrap pack at /tinfoil/mpk/mpk-<root hash>) and
hashed at boot under dirhash-sha256/1, as the inference demo hashes its base. A verifier POSTs a fresh nonce; the
service derives the secret tokens and layers from it, fingerprints the model against every listed model, and
returns one signed statement: an in-toto Statement v1 in a DSSE envelope, signed with the Tinfoil attested key,
whose `scannedModel` uses the inference statement's component shape. The statement says match / no match, which
entry, and whether the refusal-ablation signature is present. No scores, tokens or weights leave the enclave.

  GET  /health
  GET  /v1/model       the scanned model's identity, computed at boot
  GET  /v1/policy      Azure only: the launch policy (base64) whose hash the hardware report carries
  POST /v1/challenge   {"nonce": "<32-128 hex>"}  ->  {"envelope": <DSSE>, "publicKey": "<hex>"}
                       With "wait": false it answers 202 at once; the statement is then fetched with
  GET  /v1/result?nonce=<the nonce>   (202 while the check runs). A check of a large model takes minutes, and a
                       connection left idle that long is often cut on the way.
  POST /v1/generate    MODE=light with SERVE=1, after a Green light: {"nonce", "prompt", "maxTokens"}  ->
                       {"text", "envelope", "publicKey"}. The box answers with the model it has just checked.
                       Before the answer leaves, the known model is asked of every token whether it would have
                       chosen it. An answer it would not have written is withheld ("text": null), the statement
                       says so, and the box stops serving. It remembers: until it is restarted, every later
                       challenge of this model is Amber and says how many answers were withheld.

Settings (pinned in tinfoil-config.yml, so measured):
  MODEL_DIR, MODEL_NAME, MODEL_REF (repo@revision), MODEL_SCHEMA (modelwrap pack schema)
  CONCERN_LIST (path), CONCERN_LIST_DIGEST (jcs-sha256/1 of it; the service refuses to start on a mismatch)
  TOKENS (512), LAYERS (4), PORT (8080), ALLOW_DEV_KEY (1 outside an enclave: statements say keyBinding null)
  PLATFORM (tinfoil or azure-aci), ATTEST_URL (the sidecar, azure-aci only)
  OUTPUTS (1: Green also needs the model's outputs to agree with the known model's; runs both models)
  SERVE (1: with OUTPUTS, answer prompts with the checked model, test each answer against the known model's own
         choices and sign what was served)
"""
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lineage_check import fingerprint as fp4
from lineage_check import fourpillars as FP
from lineage_check import light as LT
from lineage_check import protocol as LC
from lineage_check import remote as R
from lineage_check import snp as SNP

PREDICATE_TYPE = "urn:lineage-check:LineageStatement:v0.2"
LIGHT_TYPE = "urn:lineage-check:LightStatement:v0.6"   # v0.5 adds the output check, v0.6 what was served since boot
SERVED_TYPE = "urn:lineage-check:ServedStatement:v0.3"   # v0.3 tests each answer's tokens against the known model's own choices
OUTPUTS = os.environ.get("OUTPUTS") == "1"         # Green also needs the outputs to agree with the known model's
SERVE = OUTPUTS and os.environ.get("SERVE") == "1"  # then answer prompts with the model that was checked
GEOMETRY = os.environ.get("GEOMETRY", "1") == "1"  # also match tokens by geometry when no listed model matches by name
MODE = os.environ.get("MODE", "match")
NONCE = re.compile(r"[0-9a-f]{32,128}")
DEBUG = os.environ.get("DEBUG") == "1"  # local twin only: per-entry distances to stderr (pinned off in the enclave)
PLATFORM = os.environ.get("PLATFORM", "tinfoil")
PLATFORM_NAMES = {"tinfoil": "tinfoil-containers", "azure-aci": "azure-confidential-containers"}
torch.set_grad_enabled(False)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fetch_model(ref, dest):
    """No mounted pack on Azure: download the pinned revision inside the enclave. The files are hashed after."""
    from huggingface_hub import snapshot_download
    repo, _, revision = ref.partition("@")
    snapshot_download(repo, revision=revision, local_dir=str(dest))
    shutil.rmtree(Path(dest) / ".cache", ignore_errors=True)       # download bookkeeping, not part of the model


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


def signing_key(env):
    """The key that signs statements, and whether hardware vouches for it."""
    if PLATFORM == "azure-aci":
        return Ed25519PrivateKey.generate(), True      # made here, never written out; bound by a report per statement
    return FP.load_signing_key(Path(env.get("SIGNING_KEY_PATH", FP.ATTESTED_KEY_PATH)), env.get("ALLOW_DEV_KEY") == "1")


def key_binding(pub, nonce, attested):
    if not attested:
        return None
    if PLATFORM == "azure-aci":
        return SNP.aci_attest(SNP.binding_payload(pub, nonce))
    return FP.fetch_key_binding(pub, FP.jcs_sha256({"lineageChallenge": nonce}))


class LightService:
    """MODE=light: the traffic light for one model at a pinned revision, read by byte range inside the enclave."""

    def __init__(self, env=os.environ):
        self.dir = Path(env["MODEL_DIR"]) if env.get("MODEL_DIR") else None
        repo, _, revision = env.get("MODEL_REF", "").partition("@")
        if not self.dir and not revision:
            raise RuntimeError("MODEL_REF must pin a revision: repo@commit")
        self.tokens = int(env.get("TOKENS", LT.TOKENS))
        self.key, self.attested = signing_key(env)
        self.lists = {}
        for name, var in (("watchlist", "WATCHLIST"), ("registry", "KNOWN_LIST")):
            doc = json.load(open(env[var]))
            digest = FP.jcs_sha256(doc)
            if env.get(var + "_DIGEST") not in (None, digest):
                raise RuntimeError(f"{name} digest {digest} does not match the pinned {env[var + '_DIGEST']}")
            if not all(e.get("revision") for e in doc["models"]):
                raise ValueError(f"every {name} entry needs a pinned revision")
            self.lists[name] = {"doc": doc, "digest": digest, "sources": [(e, R.Remote(e["repo"], e["revision"])) for e in doc["models"]]}
        if self.dir:                                       # a private model: read from the mount, hashed as it stands
            self.name, self.dirhash = env.get("MODEL_NAME") or self.dir.name, FP.dirhash_sha256(self.dir)
            self.q = R.LocalST(str(self.dir), label=self.name)
        else:
            self.q = R.Remote(repo, revision, file=env.get("MODEL_FILE") or None)   # MODEL_FILE: which GGUF file of the repo
        self.gguf = getattr(self.q, "gguf", None)
        self.released, self.strayed, self.tail = 0, 0, []  # answers released and withheld since boot; gaps of the last tokens served
        self.started_at = now()
        self.used, self.lock = set(), threading.Lock()
        self.held, self.last = {}, None                    # the checked model, kept for serving; the last light
        self.results = {}                                  # nonce -> (status, body) of a challenge started without waiting

    def component(self):
        if self.dir:
            return {"id": "scanned", "role": "base", "name": self.name, "scheme": FP.DIRHASH, "digest": self.dirhash, "alsoKnownAs": []}
        return {"id": "scanned", "role": "base", "name": self.q.repo, "scheme": FP.HF_REVISION,
                "ref": f"{self.q.repo}@{self.q.sha}", "alsoKnownAs": []}

    def subject(self):
        c = self.component()
        return [{"name": c["name"], "digest": {"sha256": c["digest"]} if self.dir else {"gitCommit": self.q.sha}}]

    def info(self):
        return {"model": self.component(), **{k: {"digest": v["digest"]} for k, v in self.lists.items()}}

    def ready(self):
        what = f"{self.name} dirhash {self.dirhash[:16]}..." if self.dir else f"{self.q.repo}@{self.q.sha[:10]}{' [' + self.gguf[0] + ']' if self.gguf else ''}"
        return (f"{what}, {len(self.lists['watchlist']['sources'])} listed and "
                f"{len(self.lists['registry']['sources'])} known models")

    def challenge(self, nonce):
        if not NONCE.fullmatch(nonce or ""):
            raise ValueError("the nonce must be 32 to 128 lowercase hex characters")
        with self.lock:
            if nonce in self.used:
                raise ValueError("nonce already used")
            self.used.add(nonce)
            t0 = time.time()
            check = None
            if OUTPUTS:                                    # run the loaded model and the known one on text from the nonce
                from lineage_check import outputs as O
                self.held.clear()
                check = lambda entry: O.check(self.q, entry["repo"], nonce, entry["revision"], hold=self.held)
            r = LT.light(self.q, self.lists["watchlist"]["sources"], self.lists["registry"]["sources"], nonce, self.tokens,
                         ablation=LT.ablation_signature(self.q, nonce, self.tokens), geometry=GEOMETRY, outputs=check)
            seconds = round(time.time() - t0, 1)
            if self.strayed and r["light"] == "green":     # this box has already served an answer the known model would
                r = dict(r, light="amber", matched=None, finding="strayed")       # not have written: no Green until a restart
            if r["light"] != "green":
                self.held.clear()                          # only a Green model is kept for serving
        if DEBUG:
            print(f"  {r['light']} {r['matched']} {r['distances']}", file=sys.stderr, flush=True)
        pub = FP.raw_public_key(self.key)
        binding = key_binding(pub, nonce, self.attested)
        described = lambda k: {"name": self.lists[k]["doc"]["name"], "scheme": FP.JCS, "digest": self.lists[k]["digest"],
                               "entries": len(self.lists[k]["sources"])}
        statement = {
            "_type": FP.STATEMENT_TYPE,
            "subject": self.subject(),
            "predicateType": LIGHT_TYPE,
            "predicate": {
                "scannedModel": self.component(),
                "watchlist": described("watchlist"), "registry": described("registry"),
                "challenge": {"nonce": nonce},
                "light": {"result": r["light"], "matched": r["matched"], "finding": r["finding"]},
                "method": {"name": "design-free token tables", "tokens": self.tokens, "layersSampled": LT.LAYERS,
                           "tokensByGeometry": GEOMETRY, "cutLayersRead": LT.CUT_LAYERS,
                           **({"fileRead": self.gguf[0], "shards": len(self.gguf)} if self.gguf else {})},
                "execution": {"platform": PLATFORM_NAMES[PLATFORM] if binding else "local", "startedAt": self.started_at,
                              "issuedAt": now(), "scanSeconds": seconds, "statementKey": pub.hex(),
                              "attestation": {"keyBinding": binding}},
            },
        }
        if SERVE:
            statement["predicate"]["served"] = {"released": self.released, "withheld": self.strayed}     # since boot
        if OUTPUTS:
            from lineage_check import outputs as O
            seen = r.get("outputs", "absent")
            statement["predicate"]["light"]["outputs"] = ("not run" if seen == "absent" else "not compared" if seen is None
                                                          else "agree" if seen["agree"] else "differ")
            statement["predicate"]["method"]["outputs"] = {"bar": O.BAR, "windows": O.WINDOWS, "tokens": O.LENGTH,
                                                           "textPool": f"{O.POOL[0]}@{O.POOL[2]}"}
        self.last = {"light": r["light"], "matched": r["matched"], "digest": FP.jcs_sha256(statement)}
        return {"envelope": FP.sign_envelope(statement, self.key), "publicKey": pub.hex()}

    def begin(self, nonce):
        """Start a challenge and return at once. The answer is kept for `result`."""
        if not NONCE.fullmatch(nonce or ""):
            raise ValueError("the nonce must be 32 to 128 lowercase hex characters")
        with self.lock:
            if nonce in self.used or nonce in self.results:
                raise ValueError("nonce already used")
            self.results[nonce] = None

        def work():
            try:
                self.results[nonce] = (200, self.challenge(nonce))
            except ValueError as e:
                self.results[nonce] = (400, {"error": str(e)})
            except Exception as e:  # a scan failure is an error, never a verdict
                self.results[nonce] = (500, {"error": f"{type(e).__name__}: {e}"})
        threading.Thread(target=work, daemon=True).start()
        return {"accepted": nonce}

    def result(self, nonce):
        if nonce not in self.results:
            return 404, {"error": "no challenge was started with this nonce"}
        return self.results[nonce] or (202, {"status": "running"})

    def generate(self, body):
        """Answer a prompt with the model this box has just checked, and sign what was served. One box and one key
        do both, so the model that answers is the model the light was given for. Text drawn at load time cannot
        hold a trigger only the operator knows, so the answer is tested before it leaves: the known model reads the
        prompt and the answer and says, token by token, how far each falls short of its own first choice. The budget
        runs over the last tokens served, whoever asked for them. An answer the known model would not have written
        is withheld, nothing about its text is signed, and the box stops serving and remembers."""
        nonce, prompt = body.get("nonce"), str(body.get("prompt") or "")
        if not SERVE:
            raise ValueError("this service does not serve (SERVE is off)")
        if not NONCE.fullmatch(nonce or ""):
            raise ValueError("the nonce must be 32 to 128 lowercase hex characters")
        if not prompt or len(prompt) > 4000:
            raise ValueError("the prompt must be 1 to 4,000 characters")
        with self.lock:
            if nonce in self.used:
                raise ValueError("nonce already used")
            if not self.held.get("model") or not self.last or self.last["light"] != "green":
                raise ValueError("no checked model to serve: ask for a challenge first; only a Green model is served")
            self.used.add(nonce)
            tok, model = self.held["tokenizer"], self.held["model"]
            enc = tok(prompt, return_tensors="pt")
            limit = max(1, min(int(body.get("maxTokens") or 32), 64))
            t0 = time.time()
            out = model.generate(**enc, max_new_tokens=limit, do_sample=False, pad_token_id=tok.eos_token_id)
            start, t1 = enc["input_ids"].shape[1], time.time()
            text = tok.decode(out[0, start:], skip_special_tokens=True)
            from lineage_check import outputs as O
            gap = O.answer_gap(self.held["reference"], out, start)[0]
            agrees = O.answer_passes(gap, self.tail)
            if agrees:
                measured = {"tokens": int(len(gap)), "notFirstChoice": int((gap > 0).sum()), "largestGap": round(float(gap.max()), 3),
                            "worstWindow": round(O.worst_window(gap, self.tail), 3)}
                took = {"generateSeconds": round(t1 - t0, 2), "testSeconds": round(time.time() - t1, 2)}
                self.tail, self.released = (self.tail + [float(v) for v in gap])[-(O.WINDOW - 1):], self.released + 1
            else:                                          # of a withheld text nothing is signed: no hash, no measurement,
                measured, took = {}, {}                    # no timing. The light no longer stands, and the box remembers
                self.strayed += 1
                self.held.clear()
        sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
        pub = FP.raw_public_key(self.key)
        binding = key_binding(pub, nonce, self.attested)
        statement = {
            "_type": FP.STATEMENT_TYPE,
            "subject": self.subject(),
            "predicateType": SERVED_TYPE,
            "predicate": {
                "servedModel": self.component(),
                "check": {"statement": self.last["digest"], "light": self.last["light"], "matched": self.last["matched"]},
                "request": {"nonce": nonce, "promptSha256": sha(prompt), "maxTokens": limit,
                            **({"completionSha256": sha(text)} if agrees else {})},
                "test": {"against": self.last["matched"], "budget": O.ANSWER_BUDGET, "window": O.WINDOW, "cap": O.TOKEN_CAP,
                         "result": "its own answer" if agrees else "not its answer", **measured},
                "released": agrees, "serving": "continues" if agrees else "stopped",
                "execution": {"platform": PLATFORM_NAMES[PLATFORM] if binding else "local", "issuedAt": now(), **took,
                              "statementKey": pub.hex(), "attestation": {"keyBinding": binding}},
            },
        }
        return {"text": text if agrees else None, "envelope": FP.sign_envelope(statement, self.key), "publicKey": pub.hex()}


class Service:
    def __init__(self, env=os.environ):
        self.model_dir = Path(env["MODEL_DIR"])
        self.model_name = env.get("MODEL_NAME") or self.model_dir.name
        self.model_ref = env.get("MODEL_REF") or None
        self.schema = env.get("MODEL_SCHEMA", "2")
        self.n, self.k = int(env.get("TOKENS", 512)), int(env.get("LAYERS", 4))
        self.key, self.attested = signing_key(env)
        if PLATFORM == "azure-aci" and not self.model_dir.exists():
            fetch_model(env["MODEL_REF"], self.model_dir)
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

    def info(self):
        return {"model": self.component(), "hashSeconds": self.hash_seconds, "concernList": {"digest": self.concern_digest}}

    def ready(self):
        return (f"{self.model_name} dirhash {self.dirhash[:16]}... ({self.hash_seconds}s), {len(self.listed)} listed models, "
                f"concern list {self.concern_digest[:16]}...")

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
        binding = key_binding(pub, nonce, self.attested)
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
                "execution": {"platform": PLATFORM_NAMES[PLATFORM] if binding else "local", "startedAt": self.started_at,
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
                return self._send(200, svc.info())
            if self.path.startswith("/v1/result?nonce=") and hasattr(svc, "result"):
                return self._send(*svc.result(self.path.split("=", 1)[1]))
            if self.path == "/v1/policy":                  # Azure: the launch policy the hardware report commits to
                f = Path(os.environ.get("UVM_SECURITY_CONTEXT_DIR", "/nonexistent")) / "security-policy-base64"
                return self._send(200, {"policy": f.read_text().strip() if f.exists() else None})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/challenge", "/v1/generate") or (self.path == "/v1/generate" and not hasattr(svc, "generate")):
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/v1/generate":
                    return self._send(200, svc.generate(body))
                if body.get("wait") is False and hasattr(svc, "begin"):
                    return self._send(202, svc.begin(body.get("nonce")))
                return self._send(200, svc.challenge(body.get("nonce")))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:  # a scan failure is an error, never a verdict
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (now(), fmt % args))

    print(f"{now()} ready: {svc.ready()}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    serve(LightService() if MODE == "light" else Service(), int(os.environ.get("PORT", 8080)))
