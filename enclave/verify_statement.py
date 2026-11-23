"""Verify a lineage statement with SASH's own code, and link it to an inference statement.

  python verify_statement.py RESPONSE.json --nonce HEX --concern concern.json
         [--model-dir DIR] [--inference-statement SASH_STATEMENT.json] [--upstream path/to/inference-demo]

With --upstream, the envelope is opened by inference-demo's own `open_envelope`, models are matched by its own
`identifiers()`, and digests are recomputed with its own `dirhash_sha256` and `jcs_sha256`; the vendored copy
(fourpillars.py) is checked against them. Exit 0 only if every requested check passes.
"""
import argparse
import json
import sys
from pathlib import Path

from lineage_check import fourpillars as FP
from lineage_check import snp as SNP

PREDICATE_TYPE = "urn:lineage-check:LineageStatement:v0.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("response")
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--concern", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--inference-statement")
    ap.add_argument("--upstream")
    ap.add_argument("--policy", help="Azure: the expected launch policy, as the base64 string in the deployment template")
    a = ap.parse_args()
    ok = True

    def check(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {what}")

    if a.upstream:
        sys.path.insert(0, str(Path(a.upstream) / "service" / "src"))
        sys.path.insert(0, str(Path(a.upstream) / "tools" / "src"))
        from inference_service import identity as up_identity
        from inference_tools.compare import identifiers, open_envelope
        who = "inference-demo's own"
    else:
        identifiers = lambda c: {(c["scheme"], c["digest"])} | {(x["scheme"], x.get("digest") or x.get("ref"))
                                                                 for x in c.get("alsoKnownAs") or []}
        up_identity = FP
        def open_envelope(env, pub):
            import base64
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            payload = base64.b64decode(env["payload"])
            Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(env["signatures"][0]["sig"]),
                                                           FP.pae(FP.PAYLOAD_TYPE, payload))
            return json.loads(payload)
        who = "the vendored"

    response = json.load(open(a.response))
    try:
        st = open_envelope(response["envelope"], bytes.fromhex(response["publicKey"]))
        check(True, f"signature verifies under the statement key ({who} open_envelope)")
    except Exception as e:
        check(False, f"signature: {e}")
        sys.exit(1)
    p = st["predicate"]
    check(st["predicateType"] == PREDICATE_TYPE, f"predicate type {st['predicateType']}")
    check(p["execution"]["statementKey"] == response["publicKey"], "the signing key is the statement key it names")
    check(p["challenge"]["nonce"] == a.nonce, "issued for this verifier's nonce (no replay)")
    concern = json.load(open(a.concern))
    check(p["concernList"]["digest"] == up_identity.jcs_sha256(concern) == FP.jcs_sha256(concern),
          f"concern list digest matches ({who} jcs_sha256 and the vendored copy agree)")
    binding = p["execution"]["attestation"]["keyBinding"]
    if binding is None:
        print("NOTE  keyBinding null: a local twin, signed with a dev key no hardware report binds")
    elif binding.get("platform") == SNP.PLATFORM_ACI:
        import base64
        policy = base64.b64decode(open(a.policy).read().strip()) if a.policy else None
        try:
            f = SNP.verify_binding(binding, bytes.fromhex(response["publicKey"]), a.nonce, policy)
            check(True, f"AMD SEV-SNP report verifies down to AMD's root ({f['root']}), debugging off")
            check(True, "the report names this statement key and this nonce (report_data)")
            if policy is None:
                print(f"NOTE  no --policy given: host_data {f['host_data'].hex()} not compared with an expected policy")
            else:
                check(True, f"launched under the expected policy: host_data {f['host_data'].hex()[:16]}...")
            print(f"NOTE  launch measurement {f['measurement'].hex()[:16]}...: Microsoft's utility VM, endorsed separately; not judged here")
        except SNP.AttestationError as e:
            check(False, f"hardware report: {e}")
    else:
        check(FP.bound_key(binding).hex() == response["publicKey"],
              "the attestation document lists the statement key (hardware checks: registry checks_tinfoil)")
    model = p["scannedModel"]
    if a.model_dir:
        d_up = up_identity.dirhash_sha256(Path(a.model_dir))
        check(model["digest"] == d_up == FP.dirhash_sha256(Path(a.model_dir)),
              f"scanned bytes recomputed: dirhash-sha256/1 {d_up[:16]}... ({who} dirhash_sha256)")
    if a.inference_statement:
        inf = json.load(open(a.inference_statement))
        served = open_envelope(inf["envelope"], bytes.fromhex(inf["publicKey"]))
        base = next(m for m in served["predicate"]["servingPipeline"]["models"] if m["role"] == "base")
        shared = sorted(s for s, _ in identifiers(model) & identifiers(base))
        check(shared, f"same model as the inference statement's served base {base['name']}: shared {', '.join(shared) or 'nothing'}")
    lin = p["lineage"]
    print(f"\nlineage: {lin['verdict']}{' -> ' + lin['matched'] if lin['matched'] else ''}"
          f"{'; refusal-ablation signature' if lin['refusalAblationSignature'] else ''}"
          f"  ({p['method']['tokens']} tokens, {p['method']['layersSampled']} layers, {p['execution']['platform']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
