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
LIGHT_TYPES = ("urn:lineage-check:LightStatement:v0.3", "urn:lineage-check:LightStatement:v0.4",    # v0.4 adds the finding,
               "urn:lineage-check:LightStatement:v0.5", "urn:lineage-check:LightStatement:v0.6")      # v0.5 the output check,
                                                                                                      # v0.6 what was served since boot
SERVED_TYPES = ("urn:lineage-check:ServedStatement:v0.1", "urn:lineage-check:ServedStatement:v0.3")   # v0.3 tests each answer
BUDGET, WINDOW, CAP = 9.0, 48, 4.0   # the bars of that test, as in lineage_check/outputs.py (a unit test keeps them equal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("response")
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--concern", help="the list a match statement was issued against")
    ap.add_argument("--watchlist", help="the watchlist a traffic-light statement was issued against")
    ap.add_argument("--known", help="the registry of known models a traffic-light statement was issued against")
    ap.add_argument("--model-dir")
    ap.add_argument("--inference-statement")
    ap.add_argument("--upstream")
    ap.add_argument("--policy", help="Azure: the expected launch policy, as the base64 string in the deployment template")
    ap.add_argument("--served", help="a /v1/generate response from the same box, to link to this light statement")
    ap.add_argument("--served-nonce", help="the nonce that /v1/generate request carried")
    ap.add_argument("--prompt", help="the prompt of that request, to check against the hash the statement signs")
    a = ap.parse_args()
    if a.served and not a.served_nonce:
        ap.error("--served needs --served-nonce: without it the served statement's replay and hardware checks cannot run")
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
    is_light = st["predicateType"] in LIGHT_TYPES
    check(st["predicateType"] in (PREDICATE_TYPE, *LIGHT_TYPES), f"predicate type {st['predicateType']}")
    check(p["execution"]["statementKey"] == response["publicKey"], "the signing key is the statement key it names")
    check(p["challenge"]["nonce"] == a.nonce, "issued for this verifier's nonce (no replay)")
    for field, path in (("watchlist", a.watchlist), ("registry", a.known)) if is_light else (("concernList", a.concern),):
        if not path:
            sys.exit(f"this statement needs the {field} file it was issued against")
        doc = json.load(open(path))
        check(p[field]["digest"] == up_identity.jcs_sha256(doc) == FP.jcs_sha256(doc),
              f"{field} digest matches ({who} jcs_sha256 and the vendored copy agree)")
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
            vm = f.get("utility_vm")
            if vm:
                check(True, f"the base system is one Microsoft signed: {vm['feed']}, version {vm['guest_svn']}, "
                            f"measurement {f['measurement'].hex()[:16]}...")
            else:
                print(f"NOTE  launch measurement {f['measurement'].hex()[:16]}...: no endorsement attached, not judged")
        except SNP.AttestationError as e:
            check(False, f"hardware report: {e}")
    else:
        check(FP.bound_key(binding).hex() == response["publicKey"],
              "the attestation document lists the statement key (hardware checks: registry checks_tinfoil)")
    model = p["scannedModel"]
    named = lambda c: c.get("ref") or f"{c['name']}, {c['scheme']} {c['digest'][:16]}..."     # a pinned revision, or a mounted model's hash
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
    if is_light:
        why = {"listed": "matches a listed model", "listed-by-geometry": "matches a listed model, tokens matched by geometry",
               "cut": "a known model with one direction cut out", "signature": "refusal-ablation signature",
               "known": "matches a known model", "none": "matches nothing listed or known",
               "unchecked": "matches a known model but cannot be compared with it matrix by matrix",
               "outputs": "the weights match a known model, the outputs do not",
               "strayed": "this box has served an answer the known model would not have written",
               "outputs-unchecked": "the weights match a known model, the outputs could not be compared"}.get(p["light"].get("finding"))
        if p.get("served"):
            print(f"NOTE  since boot this box released {p['served']['released']} answers and withheld {p['served']['withheld']}")
        if p["light"].get("outputs"):
            o = p["method"]["outputs"]
            print(f"NOTE  output check on: outputs {p['light']['outputs']} ({o['windows']} windows of {o['tokens']} tokens, bar {o['bar']})")
            if p["light"]["outputs"] == "agree" and p["light"]["result"] == "green":
                why = "weights and outputs match a known model"
        if a.served:
            import hashlib
            served = json.load(open(a.served))
            try:
                sv = open_envelope(served["envelope"], bytes.fromhex(served["publicKey"]))
                check(True, "served statement: signature verifies")
            except Exception as e:
                check(False, f"served statement: {e}")
                sys.exit(1)
            q = sv["predicate"]
            check(sv["predicateType"] in SERVED_TYPES, f"served statement: predicate type {sv['predicateType']}")
            check(served["publicKey"] == response["publicKey"], "served statement: signed by the same key as the light statement")
            check(q["check"]["statement"] == FP.jcs_sha256(st), "served statement: names this light statement by digest")
            check(q["check"]["light"] == "green" == p["light"]["result"], "served statement: the model served was Green")
            check(q["servedModel"] == p["scannedModel"], f"served statement: the model served is the model checked ({named(q['servedModel'])})")
            c = q.get("test")
            if c and q["released"]:
                check(c["result"] == "its own answer" and c["worstWindow"] <= c["budget"] and c["largestGap"] <= c["cap"],
                      f"served statement: {c['against']} was asked about each of the answer's {c['tokens']} tokens: {c['notFirstChoice']} are not "
                      f"its first choice, none falls more than {c['largestGap']} nats short of it (cap {c['cap']}), and no {c['window']} tokens "
                      f"served in a row fall more than {c['worstWindow']} short (budget {c['budget']}): {c['result']}")
            if c:
                check((c["budget"], c["window"], c["cap"]) == (BUDGET, WINDOW, CAP), "served statement: the budget, its window and the cap are the tool's")
            elif sv["predicateType"].endswith("v0.1"):
                print("NOTE  served statement: v0.1, issued before answers were tested. It does not show the answer was the known model's")
            else:
                check(False, "served statement: it carries no test of the answer")
            if a.prompt is not None:
                check(q["request"]["promptSha256"] == hashlib.sha256(a.prompt.encode()).hexdigest(), "served statement: it signs this prompt")
            if c and not q["released"]:
                check(c["result"] == "not its answer" and served["text"] is None and q["serving"] == "stopped"
                      and not {"completionSha256", "largestGap", "worstWindow", "tokens", "notFirstChoice", "generateSeconds", "testSeconds"}
                      & (set(q["request"]) | set(c) | set(q["execution"])),
                      f"served statement: {c['against']} would not have written this answer. It was withheld, nothing about its "
                      f"text is signed, and the box stopped serving")
            else:
                check(q["request"]["completionSha256"] == hashlib.sha256(served["text"].encode()).hexdigest(),
                      "served statement: the text returned is the text it signed")
            if a.served_nonce:
                check(q["request"]["nonce"] == a.served_nonce, "served statement: issued for this request's nonce")
            sb = q["execution"]["attestation"]["keyBinding"]
            if sb is None:
                print("NOTE  served statement: keyBinding null (local twin)")
            elif sb.get("platform") == SNP.PLATFORM_ACI and a.served_nonce:
                import base64
                policy = base64.b64decode(open(a.policy).read().strip()) if a.policy else None
                try:
                    SNP.verify_binding(sb, bytes.fromhex(served["publicKey"]), a.served_nonce, policy)
                    check(True, "served statement: its own hardware report verifies, same key, same policy")
                except SNP.AttestationError as e:
                    check(False, f"served statement: hardware report: {e}")
        print(f"\nlight: {p['light']['result'].upper()}{' -> ' + p['light']['matched'] if p['light']['matched'] else ''}"
              f"{' [' + why + ']' if why else ''}  ({named(model)}, "
              f"{'file ' + p['method']['fileRead'] + ', ' if p['method'].get('fileRead') else ''}"
              f"{p['method']['tokens']} tokens, {p['execution']['platform']})")
        sys.exit(0 if ok else 1)
    lin = p["lineage"]
    print(f"\nlineage: {lin['verdict']}{' -> ' + lin['matched'] if lin['matched'] else ''}"
          f"{'; refusal-ablation signature' if lin['refusalAblationSignature'] else ''}"
          f"  ({p['method']['tokens']} tokens, {p['method']['layersSampled']} layers, {p['execution']['platform']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
