"""Bigger abliteration census: Hub models found by search, scored with the reader-checked signature (forensic.py).

Groups: 'abliterated' (projection-style refusal removal, by name), 'uncensored' (usually fine-tuned, by name),
'control' (popular Instruct models whose names carry no refusal-removal keyword). 2 sampled layers + embedding rows.
Usage: python census_big.py <out.json>
"""
import json, os, sys, time, traceback
import numpy as np
from huggingface_hub import HfApi
from lineage_check import forensic
from lineage_check import remote as R

K = 2
BAD = ("gguf", "awq", "gptq", "exl", "mlx", "bnb", "4bit", "8bit", "vision", "-vl", "lora", "fp8", "int4", "int8",
       "nvfp4", "w4a16", "onnx", "image", "ocr", "flux", "audio", "embed", "reward")
REMOVAL = ("abliterat", "uncensor", "heretic", "lorablat", "decensor", "nsfw", "jailbr", "unalign", "refusal")
DONE = set()


def search(q, n, exclude_removal=False):
    api, out = HfApi(), []
    for m in api.list_models(search=q, sort="downloads", limit=600, expand=["safetensors", "gated", "downloads"]):
        st = getattr(m, "safetensors", None)
        total = getattr(st, "total", None) if st else None
        name = m.id.lower()
        if not total or total > 9.5e9 or total < 0.3e9 or getattr(m, "gated", False) or any(b in name for b in BAD):
            continue
        if exclude_removal and any(k in name for k in REMOVAL):
            continue
        if m.id in DONE:
            continue
        out.append(m.id)
        if len(out) >= n:
            break
    return out


def score(repo):
    src = R.Remote(repo)
    meta = R.build(src, layers=[])
    L = meta["L"]
    layers = sorted({int(round((j + 0.5) * L / K - 0.5)) for j in range(K)})
    W, Rd = [], []
    for i in layers:
        l = R.build(src, layers=[i])["layers"][i]
        W.append(np.concatenate([l["o"], l["down"]], 1))
        Rd.append(forensic.layer_readers(l))
        del l
    ab, k, agree, ks, rr = forensic.verdict(W + [meta["E"].T], Rd + [None])
    return {"flagged": ab, "dims": k, "agreement": agree, "k_per_writer": ks, "read_ratio": rr, "arch": meta["kind"]}


def main(out):
    prev = json.load(open("../results/census_v3.json"))
    DONE.update(r["model"] for r in prev)
    res = json.load(open(out)) if os.path.exists(out) else []
    DONE.update(r["model"] for r in res)
    plan = [("abliterated", m) for m in search("abliterated", 30)] + \
           [("uncensored", m) for m in search("uncensored", 10)] + \
           [("control", m) for m in search("Instruct", 20, exclude_removal=True)]
    print(f"plan: {len(plan)} models", flush=True)
    for group, repo in plan:
        t0 = time.time()
        try:
            rec = {"model": repo, "group": group, **score(repo), "seconds": round(time.time() - t0)}
        except Exception as e:
            rec = {"model": repo, "group": group, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        res.append(rec)
        json.dump(res, open(out, "w"), indent=1)
        if "error" in rec:
            print(f"{repo:60s} [{group:11s}] ERROR {rec['error'][:90]}", flush=True)
        else:
            print(f"{repo:60s} [{group:11s}] -> {'ABLATED' if rec['flagged'] else 'clean':7s} dead {rec['k_per_writer']} "
                  f"read {rec['read_ratio'] if rec['read_ratio'] is None else round(rec['read_ratio'], 2)} ({rec['seconds']}s)", flush=True)
    for g in ("abliterated", "uncensored", "control"):
        ok = [r for r in res if r["group"] == g and "error" not in r]
        print(f"{g:11s}: {sum(r['flagged'] for r in ok)}/{len(ok)} flagged", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
