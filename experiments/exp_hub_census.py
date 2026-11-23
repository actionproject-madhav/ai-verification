"""Checking the Hub's own lineage claims from the weights alone.

Population, drawn from the Hub API on the day of the run:
  declared   for five popular small base models, the most-downloaded models that declare each one as their base
             (fine-tune, merge, quantised), in a weight format the reader supports
  undeclared models with the same architecture tag and parameter count whose cards declare no base model
Every model is fingerprinted under one audit nonce (the protocol's nonce-chosen layers, 512 tokens, 2 layers) and
compared with all five anchors; the verdict is set against the card. One signed statement per model, as in the
operator audit; distances stay in the research record only.

Usage: python exp_hub_census.py build <population.json> [per_declared] [per_undeclared]
       python exp_hub_census.py scan  <population.json> <out.jsonl> <nonce> <shard> <n_shards>
"""
import base64, hashlib, json, os, shutil, sys, time
import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from huggingface_hub import HfApi
from lineage_check import fingerprint as fp4
from lineage_check import protocol as LC
from lineage_check import remote as R
from lineage_check import paths

torch.set_grad_enabled(False)
torch.set_num_threads(4)
N_TOK, K, THRESHOLD = 512, 2, 0.5
KEY = os.environ.get("LINEAGE_KEY", "demo/keys/sk.pem")
STATEMENTS = os.environ.get("LINEAGE_STATEMENTS", "results/hub_census_statements")
ANCHORS = [  # id, the repo the anchor's weights are read from, the base the Hub's metadata names
    {"id": "qwen2.5-0.5b-instruct", "repo": "Qwen/Qwen2.5-0.5B-Instruct", "base": "Qwen/Qwen2.5-0.5B-Instruct"},
    {"id": "qwen2.5-1.5b-instruct", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "base": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"id": "llama-3.2-1b-instruct", "repo": "unsloth/Llama-3.2-1B-Instruct", "base": "meta-llama/Llama-3.2-1B-Instruct"},
    {"id": "smollm2-1.7b-instruct", "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "base": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
    {"id": "qwen3-1.7b", "repo": "Qwen/Qwen3-1.7B", "base": "Qwen/Qwen3-1.7B"},
]
READABLE_QUANT = (None, "awq", "gptq")


def card_base(m):
    cd = m.card_data
    if cd is None:
        return None
    bm = cd.get("base_model") if isinstance(cd, dict) else getattr(cd, "base_model", None)
    return bm or None


def build_population(path, per_d=8, per_u=8):
    api = HfApi()
    skip_repos = {a["repo"] for a in ANCHORS} | {a["base"] for a in ANCHORS}
    pop, skipped, seen = [], [], set(skip_repos)
    for a in ANCHORS:
        for rel in ("finetune", "merge", "quantized"):
            got = 0
            for m in api.list_models(filter=f"base_model:{rel}:{a['base']}", sort="downloads", limit=300,
                                     expand=["safetensors", "config", "downloads"]):
                if m.id in seen:
                    continue
                seen.add(m.id)
                q = ((m.config or {}).get("quantization_config") or {}).get("quant_method")
                if not getattr(m, "safetensors", None):
                    continue                                   # GGUF, ONNX, adapters: not a safetensors model
                if q not in READABLE_QUANT:
                    skipped.append({"repo": m.id, "declared": a["id"], "relation": rel, "reason": f"quantisation {q}"})
                    continue
                pop.append({"repo": m.id, "group": "declared", "declared": a["id"], "relation": rel,
                            "downloads": m.downloads, "quant": q})
                got += 1
                if got >= per_d:
                    break
    for a in ANCHORS:                                          # same architecture and size, no base declared
        info = api.model_info(a["repo"], expand=["safetensors", "config"])
        total, tag = info.safetensors.total, info.config["model_type"]
        got = 0
        for m in api.list_models(filter=tag, sort="downloads", limit=2000, expand=["cardData", "safetensors", "downloads"]):
            st = getattr(m, "safetensors", None)
            if m.id in seen or not st or abs(st.total - total) > 0.005 * total or card_base(m):
                continue
            seen.add(m.id)
            pop.append({"repo": m.id, "group": "undeclared", "declared": None, "relation": None, "shape_of": a["id"],
                        "downloads": m.downloads, "quant": None})
            got += 1
            if got >= per_u:
                break
    json.dump({"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "anchors": ANCHORS,
               "population": pop, "skipped": skipped}, open(path, "w"), indent=1)
    print(f"{len(pop)} models ({sum(p['group'] == 'declared' for p in pop)} declared, "
          f"{sum(p['group'] == 'undeclared' for p in pop)} undeclared); {len(skipped)} skipped for format")


_ANCH, _SRC, _STR = {}, {}, {}


def anchors(nonce):
    if not _SRC:
        for a in ANCHORS:
            _SRC[a["id"]] = R.Remote(a["repo"])
            _STR[a["id"]] = fp4.select_strings(_SRC[a["id"]].vocab, nonce, N_TOK)
    return [(a, _SRC[a["id"]], _STR[a["id"]]) for a in ANCHORS]


def scan_model(repo, nonce):
    q = R.Remote(repo)
    Lq = q.config["num_hidden_layers"]
    q_layers = LC.layer_plan(Lq, nonce, K)
    dists, fq_cache, ablated = {}, {}, None
    for a, ref, strings in anchors(nonce):
        key = tuple(strings)
        if key not in fq_cache:
            fq_cache[key] = LC.fingerprint(q, strings, q_layers, forensic=ablated is None)
            ablated = fq_cache[key][0].get("ablation_signature", ablated)
        fq, kq = fq_cache[key]
        a_layers = LC.anchor_layers(ref.config["num_hidden_layers"], Lq, q_layers)
        akey = (a["id"], tuple(a_layers))
        if akey not in _ANCH:
            _ANCH[akey] = LC.fingerprint(ref, strings, a_layers)
        fr, kr = _ANCH[akey]
        common = sorted(set(kq) & set(kr))
        if len(common) < 0.5 * len(strings):
            dists[a["id"]] = None                              # a different tokenizer: not comparable
            continue
        r = fp4.compare(LC.restrict(fr, kr, common), LC.restrict(fq, kq, common), align=True)
        dists[a["id"]] = float(np.mean([r["qk"], r["ov"]]))
    return q, q_layers, dists, bool(ablated)


def sign(repo, sha, verdict, matched, ablated, nonce, sk):
    st = {"_type": "https://in-toto.io/Statement/v1", "subject": [{"name": repo, "digest": {"gitCommit": sha}}],
          "predicateType": LC.PREDICATE,
          "predicate": {"verdict": verdict, "matched": matched, "ablation_signature": ablated, "nonce": nonce,
                        "list": {"digest": {"sha256": hashlib.sha256(json.dumps(ANCHORS, sort_keys=True).encode()).hexdigest()}},
                        "method": {"name": "token-anchored circuits", "version": fp4.VERSION, "tokens": N_TOK,
                                   "layers_sampled": K, "layer_rule": LC.LAYER_RULE},
                        "issuedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    body = json.dumps(st, sort_keys=True).encode()
    env = {"payloadType": LC.PAYLOAD_TYPE, "payload": base64.b64encode(body).decode(),
           "signatures": [{"sig": base64.b64encode(sk.sign(LC.pae(LC.PAYLOAD_TYPE, body))).decode()}]}
    os.makedirs(STATEMENTS, exist_ok=True)
    json.dump(env, open(os.path.join(STATEMENTS, repo.replace("/", "__") + ".dsse.json"), "w"), indent=1)


def scan_population(pop_path, out, nonce, shard, n_shards):
    pop = json.load(open(pop_path))["population"]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["repo"] for l in open(out) if l.strip()}
    sk = serialization.load_pem_private_key(open(KEY, "rb").read(), None)
    keep = {a["repo"] for a in ANCHORS}
    for i, p in enumerate(pop):
        if i % n_shards != shard or p["repo"] in done:
            continue
        t0 = time.time()
        rec = dict(p)
        try:
            q, layers, dists, ablated = scan_model(p["repo"], nonce)
            ok = {k: v for k, v in dists.items() if v is not None}
            best = min(ok, key=ok.get) if ok else None
            matched = best if best is not None and ok[best] < THRESHOLD else None
            rec.update(sha=q.sha, layers=layers, n_layers=q.config["num_hidden_layers"], model_type=q.config["model_type"],
                       distances=dists, matched=matched, ablation_signature=ablated, seconds=round(time.time() - t0))
            sign(p["repo"], q.sha, "match" if matched else "no_match", matched, ablated, nonce, sk)
            d_s = " ".join(f"{k.split('-')[0][:5]}{v:.3f}" for k, v in ok.items())
            print(f"[{shard}] {p['repo'][:58]:58s} {p['group'][:4]} {str(p['declared'])[:22]:22s} -> {str(matched):22s} "
                  f"{'ABL' if ablated else '   '} {d_s} ({rec['seconds']}s)", flush=True)
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {str(e)[:200]}", seconds=round(time.time() - t0))
            print(f"[{shard}] {p['repo'][:58]:58s} ERROR {rec['error'][:120]}", flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if p["repo"] not in keep:                              # keep the disk bounded: drop this model's cached tensors
            shutil.rmtree(os.path.join(R.CACHE, p["repo"].replace("/", "--")), ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build_population(sys.argv[2], *[int(x) for x in sys.argv[3:5]])
    else:
        scan_population(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
