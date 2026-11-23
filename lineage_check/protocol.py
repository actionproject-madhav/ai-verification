"""lineage-check v0: a one-bit, signed answer to "does this deployment derive from a listed model?"

Protocol (commit, then challenge):
  1. The operator commits to a weights revision (a Hub commit here; a Modelwrap root hash in an enclave).
  2. The verifier issues a fresh random nonce, never reused.
  3. The scanner derives a secret token set and layer sample from the nonce, fingerprints the committed
     weights and every listed model, and signs an in-toto Statement in a DSSE envelope. The statement says
     match / no match and which entry; it carries no scores and no tokens.
The Ed25519 key stands in for an enclave's attested key (Tinfoil attested keys work the same way).

  lineage-check keygen --dir keys
  lineage-check scan   --query REPO --commit SHA --list watchlist.json --nonce HEX --key keys/sk.pem --out st.json
  lineage-check verify --statement st.json --pub keys/pk.pem --nonce HEX
  lineage-check light  --query REPO --watchlist watchlists/government-named.json [--known known.json]
"""
import argparse, base64, hashlib, json, os, secrets, sys, time
import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from . import fingerprint as fp4
from . import remote as R

PREDICATE = "urn:lineage-check:statement:v0.1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"


def pae(t, body):
    return b"DSSEv1 %d %s %d %s" % (len(t), t.encode(), len(body), body)


LAYER_RULE = "nonce"


def sign(statement, sk):
    """A DSSE envelope over the statement, signed with an Ed25519 key."""
    body = json.dumps(statement, sort_keys=True).encode()
    pk_raw = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {"payloadType": PAYLOAD_TYPE, "payload": base64.b64encode(body).decode(),
            "signatures": [{"keyid": hashlib.sha256(pk_raw).hexdigest()[:16],
                            "sig": base64.b64encode(sk.sign(pae(PAYLOAD_TYPE, body))).decode()}]}


def open_envelope(env, pk, nonce=None):
    """The statement inside the envelope, once its signature verifies and (if given) the nonce matches."""
    body = base64.b64decode(env["payload"])
    try:
        pk.verify(base64.b64decode(env["signatures"][0]["sig"]), pae(env["payloadType"], body))
    except Exception as e:
        raise ValueError("invalid signature") from e
    st = json.loads(body)
    if nonce and st["predicate"]["nonce"] != nonce:
        raise ValueError("statement was issued for a different nonce (replay)")
    return st


def layer_plan(L, nonce, k):
    """k layers chosen by the nonce (the operator cannot know which layers are read). LAYER_RULE = 'even' takes k
    evenly spaced layers instead; the toy audit uses it to reuse cached layers, and says so."""
    if L <= k:
        return list(range(L))
    if LAYER_RULE == "even":
        return sorted({int(round((j + 0.5) * L / k - 0.5)) for j in range(k)})
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("layers" + nonce).encode()).digest()[:8], "little"))
    return sorted(rng.choice(L, size=k, replace=False).tolist())


def anchor_layers(L_a, L_q, q_layers):
    """Same depth: the same layers. Different depth (padding, pruning): each relative position and its neighbours."""
    if L_a == L_q:
        return list(q_layers)
    out = set()
    for j in q_layers:
        c = int(round(j * (L_a - 1) / max(1, L_q - 1)))
        out |= {x for x in (c - 1, c, c + 1) if 0 <= x < L_a}
    return sorted(out)


def _cut(W):
    ev, vec = np.linalg.eigh(W @ W.T)
    return float(ev[0] / max(ev[1], 1e-30)), vec[:, 0]


def fingerprint(src, strings, layers, forensic=False):
    m = R.build(src, layers=[])
    rows = fp4.rows_for(m, src.vocab, strings)
    keep = [i for i, r in enumerate(rows) if r is not None]
    E = torch.from_numpy(np.ascontiguousarray(m["E"]))[torch.as_tensor([rows[i] for i in keep])]
    fp = fp4.fingerprint(m, [rows[i] for i in keep], layers=[])
    writers, readers = [], []
    for i in layers:
        mi = R.build(src, layers=[i])
        l = mi["layers"][i]
        fp["layers"][i] = fp4.layer_comps(l, E, mi)
        if forensic:
            from . import forensic as F
            down = l["down"] if l.get("down") is not None else np.concatenate([e[2] for e in l["experts"]], 1)
            writers.append(np.concatenate([l["o"], down], 1))          # mixture of experts: every expert writes
            readers.append(F.layer_readers(l))
        del mi
    if forensic:                                   # refusal-ablation signature: a small subspace no writer writes to
        from . import forensic as F
        fp["ablation_signature"] = F.verdict(writers + [m["E"].T], readers + [None])[0]
    return fp, keep


def restrict(fp, keep_self, common):
    pos = {t: k for k, t in enumerate(keep_self)}
    idx = torch.as_tensor([pos[t] for t in common])
    sub = lambda M: M[idx][:, idx]
    return {"EE": sub(fp["EE"]), "EU": sub(fp["EU"]),
            "layers": {i: {c: sub(M) for c, M in L.items()} for i, L in fp["layers"].items()}}


_ANCHORS = {}


def scan_one(query, commit, lst, nonce, n, k, sk, out_path, debug=False):
    list_digest = hashlib.sha256(json.dumps(lst, sort_keys=True).encode()).hexdigest()
    q = R.Remote(query, commit)
    if commit and not q.sha.startswith(commit):
        sys.exit(f"committed revision {commit} does not match {q.sha}")
    Lq = q.config["num_hidden_layers"]
    q_layers = layer_plan(Lq, nonce, k)
    verdict, matched, t0, fq_cache, ablated = "no_match", None, time.time(), {}, None
    for e in lst["models"]:
        ref = R.Remote(e["repo"], e.get("revision"))
        strings = fp4.select_strings(ref.vocab, nonce, n)
        key_q = tuple(strings)
        if key_q not in fq_cache:
            fq_cache[key_q] = fingerprint(q, strings, q_layers, forensic=ablated is None)
            ablated = fq_cache[key_q][0].get("ablation_signature", ablated)
        fq, kq = fq_cache[key_q]
        a_layers = anchor_layers(ref.config["num_hidden_layers"], Lq, q_layers)
        akey = (e["id"], nonce, n, tuple(a_layers))
        if akey not in _ANCHORS:
            _ANCHORS[akey] = fingerprint(ref, strings, a_layers)
        fr, kr = _ANCHORS[akey]
        common = sorted(set(kq) & set(kr))
        if len(common) < 0.5 * len(strings):
            continue                                  # different tokenizer: not comparable
        r = fp4.compare(restrict(fr, kr, common), restrict(fq, kq, common), align=True)
        d_attn = float(np.mean([r["qk"], r["ov"]]))
        if debug:
            print(f"  {e['id']}: attention-circuit distance {d_attn:.4f} (threshold {e['threshold']})", file=sys.stderr)
        if d_attn < e["threshold"]:
            verdict, matched = "match", e["id"]
            break
    statement = {"_type": "https://in-toto.io/Statement/v1",
                 "subject": [{"name": query, "digest": {"gitCommit": q.sha}}],
                 "predicateType": PREDICATE,
                 "predicate": {"verdict": verdict, "matched": matched, "ablation_signature": bool(ablated),
                               "nonce": nonce, "list": {"digest": {"sha256": list_digest}},
                               "method": {"name": "token-anchored circuits", "version": fp4.VERSION,
                                          "tokens": n, "layers_sampled": len(q_layers), "layer_rule": LAYER_RULE},
                               "issuedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    json.dump(sign(statement, sk), open(out_path, "w"), indent=1)
    return {"query": query, "commit": q.sha[:10], "verdict": verdict, "matched": matched,
            "ablation_signature": bool(ablated), "seconds": round(time.time() - t0)}


def _key(path):
    return serialization.load_pem_private_key(open(path, "rb").read(), None)


def scan(a):
    torch.set_grad_enabled(False)
    r = scan_one(a.query, a.commit, json.load(open(a.list)), a.nonce, a.n, a.k, _key(a.key), a.out, a.debug)
    print(f"{r['verdict']}{' (' + r['matched'] + ')' if r['matched'] else ''}"
          f"{'  [refusal-ablation signature]' if r['ablation_signature'] else ''}  [{r['seconds']}s] -> {a.out}")


def scan_batch(a):
    """One audit session: one nonce, many deployments, one signed statement each, plus an operator report."""
    torch.set_grad_enabled(False)
    lst, sk = json.load(open(a.list)), _key(a.key)
    os.makedirs(a.outdir, exist_ok=True)
    report = []
    for query in [x.strip() for x in open(a.queries) if x.strip() and not x.startswith("#")]:
        out = os.path.join(a.outdir, query.replace("/", "__") + ".dsse.json")
        try:
            r = scan_one(query, None, lst, a.nonce, a.n, a.k, sk, out, a.debug)
        except Exception as e:
            r = {"query": query, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        report.append(r)
        json.dump({"nonce": a.nonce, "list": lst["name"], "results": report}, open(os.path.join(a.outdir, "report.json"), "w"), indent=1)
        if "error" in r:
            print(f"{query:56s} ERROR {r['error']}", flush=True)
        else:
            print(f"{query:56s} {r['verdict']:8s} {r['matched'] or '':22s} {'ABLATED' if r['ablation_signature'] else '':8s} {r['seconds']:>4}s", flush=True)


def verify(a):
    pk = serialization.load_pem_public_key(open(a.pub, "rb").read())
    try:
        st = open_envelope(json.load(open(a.statement)), pk, a.nonce)
    except ValueError as e:
        sys.exit(f"INVALID: {e}")
    p = st["predicate"]
    print(f"valid signature; subject {st['subject'][0]['name']}@{st['subject'][0]['digest']['gitCommit'][:10]}; "
          f"verdict {p['verdict']}{' -> ' + p['matched'] if p['matched'] else ''}"
          f"{'; refusal-ablation signature' if p.get('ablation_signature') else ''}")


def light(a):
    """Red, Green or Amber for one model: any architecture, no key needed."""
    from . import light as LT
    torch.set_grad_enabled(False)
    src = R.Remote(a.query, a.commit, file=a.file)
    nonce = a.nonce or secrets.token_hex(16)
    check = None
    if a.outputs:                                        # Green also needs the outputs to agree with the known model's
        from . import outputs as O
        check = lambda entry: O.check(src, entry["repo"], nonce, entry.get("revision"))
    out = LT.light(src, LT.sources(a.watchlist), LT.sources(a.known) if a.known else [], nonce, a.n,
                   ablation=None if a.no_ablation else LT.ablation_signature(src, nonce, a.n), geometry=a.geometry, outputs=check)
    if a.json:
        print(json.dumps({"query": a.query, "commit": src.sha, "file": src.gguf[0] if src.gguf else None, "nonce": nonce, **out}))
        return
    if src.gguf:
        print(f"read from {src.gguf[0]}" + (f" and {len(src.gguf) - 1} more shards" if len(src.gguf) > 1 else ""))
    print(f"{out['light'].upper()}{'  ' + out['matched'] if out['matched'] else ''}  ({out['reason']})")
    for name, d in out["distances"].items():
        print(f"  {name:24s} " + ("not comparable" if d is None else f"closest part {max(d[0], 0):.3f}   all parts {max(d[1], 0):.3f}"))
    if out.get("outputs"):
        o = out["outputs"]
        print(f"  outputs                  {o['mean']:.3f} from the known model's over {o['windows']} windows (agree at {o['bar']:.2f} or less)")


def keygen(a):
    os.makedirs(a.dir, exist_ok=True)
    sk = Ed25519PrivateKey.generate()
    open(os.path.join(a.dir, "sk.pem"), "wb").write(sk.private_bytes(serialization.Encoding.PEM,
                                                    serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    open(os.path.join(a.dir, "pk.pem"), "wb").write(sk.public_key().public_bytes(serialization.Encoding.PEM,
                                                    serialization.PublicFormat.SubjectPublicKeyInfo))
    print("keys written to", a.dir)


def main(argv=None):
    global LAYER_RULE
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    k = sp.add_parser("keygen"); k.add_argument("--dir", required=True)
    s = sp.add_parser("scan")
    for x in ("--query", "--list", "--nonce", "--key", "--out"):
        s.add_argument(x, required=True)
    s.add_argument("--commit"); s.add_argument("--n", type=int, default=256); s.add_argument("--k", type=int, default=4)
    s.add_argument("--debug", action="store_true")
    b = sp.add_parser("scan-batch")
    for x in ("--queries", "--list", "--nonce", "--key", "--outdir"):
        b.add_argument(x, required=True)
    b.add_argument("--n", type=int, default=256); b.add_argument("--k", type=int, default=2)
    b.add_argument("--debug", action="store_true")
    b.add_argument("--layer-rule", choices=["nonce", "even"], default="nonce")
    t = sp.add_parser("light")
    t.add_argument("--query", required=True); t.add_argument("--watchlist", required=True)
    t.add_argument("--known"); t.add_argument("--commit"); t.add_argument("--nonce")
    t.add_argument("--n", type=int, default=256); t.add_argument("--json", action="store_true")
    t.add_argument("--no-ablation", action="store_true")
    t.add_argument("--file", help='in a repository of GGUF files, the one to read, by part of its name ("Q8_0")')
    t.add_argument("--outputs", action="store_true",
                   help="Green also needs the model's outputs to agree with the known model's (runs both models)")
    t.add_argument("--geometry", action="store_true",
                   help="if no listed model matches by token name, match tokens by geometry too (reads whole token tables)")
    v = sp.add_parser("verify")
    v.add_argument("--statement", required=True); v.add_argument("--pub", required=True); v.add_argument("--nonce")
    a = ap.parse_args(argv)
    LAYER_RULE = getattr(a, "layer_rule", "nonce")
    {"keygen": keygen, "scan": scan, "scan-batch": scan_batch, "verify": verify, "light": light}[a.cmd](a)


if __name__ == "__main__":
    main()
