"""Scan the most-downloaded text-generation models on the Hub and ask where each one comes from.

Three steps, each resumable:

  build    list the top models the reader can parse, with the base model each card declares, and derive the
           registry of known base models: every repo that at least MIN_REFS listed models name as their base
  scan     fingerprint every listed model and every registry model once: two layers at fixed relative depths,
           256 tokens drawn from one nonce per tokenizer, plus the refusal-ablation signature (which reads every
           layer, so only for models up to FORENSIC_MAX parameters)
  compare  nearest registry model for every scanned model; agreement with the card; the share of models that are
           close copies (< 0.1), modified copies (< 0.5) or of unknown origin

This is a census, so it uses fixed relative depths and one nonce instead of a fresh challenge per model. Models
that share a tokenizer share the token draw, which makes their fingerprints directly comparable.

Run the scan in passes (SCAN_PASS=<name>): many workers first, then a few workers for whatever failed. A worker that
the system kills for memory loses only the model it was reading; restart it and it carries on with the next one.

Usage: python exp_hub_scan.py build   <dir> [top_n]
       python exp_hub_scan.py scan    <dir> <nonce> <shard> <n_shards>
       python exp_hub_scan.py compare <dir>
"""
import collections, hashlib, json, os, shutil, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one worker must not take every core
    os.environ.setdefault(_v, os.environ.get("SCAN_THREADS", "2"))
import numpy as np
import torch
from lineage_check import fingerprint as fp4
from lineage_check import protocol as LC
from lineage_check import remote as R

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("SCAN_THREADS", 2)))
N_TOK, K_LAYERS, MIN_REFS = 256, 2, int(os.environ.get("MIN_REFS", 10))
READABLE_QUANT = (None, "awq", "gptq")
MAX_PARAMS, MIN_PARAMS = 130e9, 50e6
COMPS = ("qk", "ov", "mlp")
FORENSIC_MAX = float(os.environ.get("FORENSIC_MAX", 16e9))    # the refusal-ablation check reads every layer: cap its size
PASS = os.environ.get("SCAN_PASS", "")                       # a failed model is retried in a later pass, not on every restart


def _bases(m):
    """Declared base models and relations, from card metadata and tags."""
    cd = m.card_data
    b = (cd.get("base_model") if isinstance(cd, dict) else getattr(cd, "base_model", None)) if cd is not None else None
    bases = [b] if isinstance(b, str) else list(b or [])
    rel = collections.defaultdict(set)
    for t in m.tags or []:
        if t.startswith("base_model:"):
            parts = t.split(":", 2)
            if len(parts) == 3 and parts[1] in ("finetune", "quantized", "merge", "adapter"):
                rel[parts[2]].add(parts[1])
    for x in rel:
        if x not in bases:
            bases.append(x)
    return [{"repo": x, "relations": sorted(rel.get(x, []))} for x in bases]


def build(d, top_n=3000):
    from huggingface_hub import HfApi
    api = HfApi()
    os.makedirs(d, exist_ok=True)
    seen, pop, skipped, refs = set(), [], collections.Counter(), collections.Counter()
    it = api.list_models(pipeline_tag="text-generation", sort="downloads", limit=40000,
                         expand=["safetensors", "cardData", "downloads", "config", "tags", "gated"])
    for m in it:
        if m.id in seen:
            continue
        seen.add(m.id)
        cfg = m.config or {}
        bases = _bases(m)
        for b in bases:                                          # every listed model votes for its base
            if "adapter" not in b["relations"]:
                refs[b["repo"]] += 1
        if len(pop) >= top_n:
            continue
        st = getattr(m, "safetensors", None)
        if cfg.get("model_type") not in R.SUPPORTED:
            skipped["architecture " + str(cfg.get("model_type"))] += 1
            continue
        if not st:
            skipped["no safetensors"] += 1
            continue
        q = (cfg.get("quantization_config") or {}).get("quant_method")
        if q not in READABLE_QUANT:
            skipped[f"quantisation {q}"] += 1
            continue
        if not (MIN_PARAMS <= st.total <= MAX_PARAMS) and q is None:
            skipped["size"] += 1
            continue
        if any("adapter" in b["relations"] for b in bases):
            skipped["adapter"] += 1
            continue
        if m.gated:
            skipped["gated"] += 1
            continue
        pop.append({"repo": m.id, "downloads": m.downloads, "params": st.total, "model_type": cfg.get("model_type"),
                    "quant": q, "declared": bases})
    in_pop = {p["repo"] for p in pop}
    registry = [{"repo": r, "refs": n} for r, n in refs.most_common() if n >= MIN_REFS]
    json.dump({"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "listed": len(seen), "population": pop,
               "registry": registry, "skipped": dict(skipped.most_common())}, open(os.path.join(d, "population.json"), "w"), indent=1)
    print(f"listed {len(seen)}; population {len(pop)}; registry {len(registry)} bases named by >= {MIN_REFS} models "
          f"({sum(r['repo'] in in_pop for r in registry)} already in the population)")
    print("skipped:", dict(skipped.most_common(8)))


def _std(M):
    M = M.double()
    M = M - M.mean()
    s = M.norm() / M.numel() ** 0.5
    return (M / s).to(torch.float16).numpy() if s > 0 else None


def scan_one(repo, nonce):
    src = R.Remote(repo)
    src.index.pop("lm_head.weight", None)                         # the census does not use the unembedding
    cfg = src.config
    L = cfg["num_hidden_layers"]
    LC.LAYER_RULE = "even"
    layers = LC.layer_plan(L, nonce, K_LAYERS)
    vocab = src.vocab
    pool = fp4.token_pool(vocab, R.ROW_LO, R.ROW_HI)
    group = hashlib.sha256("\n".join(pool).encode()).hexdigest()[:16]
    strings = fp4.select_strings(vocab, nonce, N_TOK)
    params = sum(int(np.prod(src.shape(n))) for n in src.index)
    forensic = params <= FORENSIC_MAX
    f, keep = LC.fingerprint(src, strings, layers, forensic=forensic)
    arrays = {}
    for j, i in enumerate(layers):
        for c in COMPS:
            if c in f["layers"][i]:
                a = _std(f["layers"][i][c])
                if a is not None:
                    arrays[f"{j}_{c}"] = a
    meta = {"sha": src.sha, "model_type": cfg["model_type"], "L": L, "d": cfg["hidden_size"], "layers": layers,
            "group": group, "tokens": len(keep), "params": params,
            "ablation_signature": bool(f.get("ablation_signature")) if forensic else None}
    return arrays, meta


def scan(d, nonce, shard, n_shards):
    P = json.load(open(os.path.join(d, "population.json")))
    repos = [p["repo"] for p in P["population"]] + [r["repo"] for r in P["registry"]]
    repos = list(dict.fromkeys(repos))
    fdir, out = os.path.join(d, "fps"), os.path.join(d, f"scan_{shard}.jsonl")
    mark = os.path.join(d, f"inflight_{shard}.txt")
    os.makedirs(fdir, exist_ok=True)

    def write(rec):
        with open(out, "a") as fh:
            fh.write(json.dumps({**rec, "pass": PASS} if PASS else rec) + "\n")

    if os.path.exists(mark):                                      # the last run of this shard died inside this model
        write({"repo": open(mark).read().strip(), "error": "killed while scanning (out of memory)", "seconds": None})
        os.remove(mark)
    done = set()                                                  # failed models are retried, once per pass
    for f in os.listdir(d):
        if f.startswith("scan_") and f.endswith(".jsonl"):
            for r in map(json.loads, filter(str.strip, open(os.path.join(d, f)))):
                if "error" not in r or (PASS and r.get("pass") == PASS):
                    done.add(r["repo"])
    for i, repo in enumerate(repos):
        if i % n_shards != shard or repo in done:
            continue
        t0 = time.time()
        rec = {"repo": repo}
        with open(mark, "w") as fh:
            fh.write(repo)
        for attempt in range(6):
            try:
                arrays, meta = scan_one(repo, nonce)
                np.savez(os.path.join(fdir, repo.replace("/", "__") + ".npz"), **arrays)
                rec.update(meta, seconds=round(time.time() - t0, 1))
            except Exception as e:
                if "429" in str(e) and attempt < 5:               # rate limited: wait, then try the same model again
                    time.sleep(120)
                    continue
                rec.update(error=f"{type(e).__name__}: {str(e)[:160]}", seconds=round(time.time() - t0, 1))
            break
        write(rec)
        os.remove(mark)
        print(f"[{shard}] {repo[:60]:60s} {rec.get('group', rec.get('error', ''))[:60]:60s} {rec['seconds']}s", flush=True)
        shutil.rmtree(os.path.join(R.CACHE, repo.replace("/", "--")), ignore_errors=True)


def compare(d):
    P = json.load(open(os.path.join(d, "population.json")))
    recs = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("scan_") and f.endswith(".jsonl"):
            for l in open(os.path.join(d, f)):
                if l.strip():
                    r = json.loads(l)
                    if "error" not in r or "error" in recs.get(r["repo"], r):    # a success is never replaced by a failure
                        recs[r["repo"]] = r
    registry = {r["repo"] for r in P["registry"]}
    groups = collections.defaultdict(list)
    for repo, r in recs.items():
        if "error" not in r:
            groups[r["group"]].append(repo)
    keys = [f"{j}_{c}" for j in range(K_LAYERS) for c in ("qk", "ov")]
    nearest = {}
    for g, members in groups.items():
        X, ok = [], []
        for repo in members:
            z = np.load(os.path.join(d, "fps", repo.replace("/", "__") + ".npz"))
            if all(k in z for k in keys):
                X.append(np.stack([z[k].astype(np.float32).ravel() for k in keys]))
                ok.append(repo)
        if not ok:
            continue
        X = np.stack(X)                                           # (models, 4 blocks, tokens^2), each block standardised
        roots = [i for i, repo in enumerate(ok) if repo in registry]
        for i, repo in enumerate(ok):
            best = (None, 2.0)
            for j in roots:
                if j == i:
                    continue
                dist = 1 - float(np.mean(np.einsum("bk,bk->b", X[i], X[j]) / X.shape[2]))
                if dist < best[1]:
                    best = (ok[j], dist)
            nearest[repo] = best
    rows, counts = [], collections.Counter()
    for p in P["population"]:
        repo, r = p["repo"], recs.get(p["repo"])
        if r is None:
            continue
        if "error" in r:
            counts["unreadable"] += 1
            rows.append({"repo": repo, "status": "unreadable", "error": r["error"]})
            continue
        base, dist = nearest.get(repo, (None, 2.0))
        declared = [b["repo"] for b in p["declared"]]
        light = "close copy" if dist < 0.1 else ("modified copy" if dist < 0.5 else "unknown origin")
        dd = None
        if declared:
            z = [nearest_to(d, recs, keys, repo, b) for b in declared]
            z = [x for x in z if x is not None]
            dd = min(z) if z else None
        claim = ("no base declared" if not declared else "declared base not scanned" if dd is None
                 else "declared base confirmed" if dd < 0.5 else "declared base not confirmed")
        counts[light] += 1
        counts[claim] += 1
        if r["ablation_signature"] is not None:
            counts["checked for refusal ablation"] += 1
            counts["refusal-ablation signature"] += bool(r["ablation_signature"])
        rows.append({"repo": repo, "downloads": p["downloads"], "nearest": base, "distance": round(dist, 4), "light": light,
                     "declared": declared, "distance_to_declared": None if dd is None else round(dd, 4), "claim": claim,
                     "ablation_signature": r["ablation_signature"]})
    json.dump({"counts": dict(counts), "rows": rows}, open(os.path.join(d, "compare.json"), "w"), indent=1)
    for k, v in counts.most_common():
        print(f"  {k:28s} {v}")


def nearest_to(d, recs, keys, a, b):
    """Distance between two scanned models that share a tokenizer, else None."""
    ra, rb = recs.get(a), recs.get(b)
    if not ra or not rb or "error" in ra or "error" in rb or ra["group"] != rb["group"]:
        return None
    za = np.load(os.path.join(d, "fps", a.replace("/", "__") + ".npz"))
    zb = np.load(os.path.join(d, "fps", b.replace("/", "__") + ".npz"))
    if not all(k in za and k in zb for k in keys):
        return None
    return 1 - float(np.mean([np.mean(za[k].astype(np.float32) * zb[k].astype(np.float32)) for k in keys]))


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(d, *[int(x) for x in sys.argv[3:4]])
    elif cmd == "scan":
        scan(d, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        compare(d)
