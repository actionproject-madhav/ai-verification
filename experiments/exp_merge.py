"""Mixed heritage: merges and frankenmerges (Casper et al. 2026 list these as unaddressed).

  soup     : 50/50 linear merge of Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct (both descend from Qwen2.5-0.5B)
  cross    : 50/50 linear merge of Qwen2.5-0.5B and Qwen2-0.5B (independent parents)
  franken  : layers 0-11 from Qwen2.5-0.5B-Instruct, layers 12-23 from Qwen2-0.5B (passthrough merge)
For each: attention-circuit distance to each candidate parent, and per-layer attribution.

Usage: python exp_merge.py <out.json>
"""
import gc, glob, json, sys
import numpy as np
import torch
from lineage_check import fingerprint as fp4
from lineage_check import remote as R
from lineage_check import paths

torch.set_grad_enabled(False)
H = paths.HUB
REPOS = {"Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", "Qwen2-0.5B": "Qwen/Qwen2-0.5B",
         "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct", "Coder-0.5B-Instruct": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}


def part(name):
    repo = REPOS[name]
    return R.build(R.LocalST(sorted(glob.glob(f"{H}/models--{repo.replace('/', '--')}/snapshots/*"))[0], repo=repo))


def lin(a, b, t=0.5):
    m = dict(a)
    m["E"] = (1 - t) * a["E"] + t * b["E"]
    m["U"] = None if a["U"] is None or b["U"] is None else (1 - t) * a["U"] + t * b["U"]
    m["final_g"] = (1 - t) * a["final_g"] + t * b["final_g"]
    m["layers"] = {i: {k: (1 - t) * v + t * b["layers"][i][k] for k, v in L.items()} for i, L in a["layers"].items()}
    return m


def attn(r):
    return float(np.mean([r["qk"], r["ov"]]))


def main(out):
    vocab = R.Remote("Qwen/Qwen2.5-0.5B").vocab
    rows = [vocab[s] - R.ROW_LO for s in fp4.select_strings(vocab, "merge-audit", 512)]
    parents = {}
    for p in ("Qwen2.5-0.5B", "Qwen2-0.5B"):
        parents[p] = fp4.fingerprint(part(p), rows)
        gc.collect()
    queries = {}
    a, b = part("Qwen2.5-0.5B-Instruct"), part("Coder-0.5B-Instruct")
    queries["soup: Instruct + Coder-Instruct (same ancestor)"] = fp4.fingerprint(lin(a, b), rows)
    del b
    gc.collect()
    q2 = part("Qwen2-0.5B")
    fr = dict(a)
    fr["layers"] = {i: (a["layers"][i] if i < 12 else q2["layers"][i]) for i in a["layers"]}
    queries["frankenmerge: Qwen2.5-Instruct layers 0-11 + Qwen2 layers 12-23"] = fp4.fingerprint(fr, rows)
    del a, fr
    gc.collect()
    base = part("Qwen2.5-0.5B")
    queries["cross merge: Qwen2.5 + Qwen2 (independent parents)"] = fp4.fingerprint(lin(base, q2), rows)
    del base, q2
    gc.collect()
    res = []
    for qn, fq in queries.items():
        rec = {"query": qn, "to": {p: attn(fp4.compare(parents[p], fq)) for p in parents}}
        att = []
        for j in sorted(fq["layers"]):
            s = {p: float(np.mean([float(fp4.corr(parents[p]["layers"][j][c], fq["layers"][j][c])) for c in ("qk", "ov")]))
                 for p in parents}
            att.append({"layer": j, "best": max(s, key=s.get), "sim": s})
        rec["layers"] = att
        res.append(rec)
        print(f"{qn}\n   distance to parents: " + ", ".join(f"{p} {v:.3f}" for p, v in rec["to"].items()))
        print("   layer attribution: " + " ".join(("A" if x["best"] == "Qwen2.5-0.5B" else "B") for x in att)
              + "   (A = Qwen2.5 lineage, B = Qwen2 lineage)", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
