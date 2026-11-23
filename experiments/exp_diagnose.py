"""Why did the census miss two abliterated releases? Diff each against its parent in the same sampled layers.

For each layer and matrix: relative size of the edit ||dW|| / ||W||, its effective rank (singular values of dW),
and which side carries it: a writer-side cut has dW = -r r^T W (left singular vector shared across matrices),
a reader-side cut has dW = -W r r^T (right singular vector shared).

Usage: python exp_diagnose.py <out.json>
"""
import json, sys
import numpy as np
from lineage_check import remote as R

PAIRS = [("failspy/Meta-Llama-3-8B-Instruct-abliterated-v3", "NousResearch/Meta-Llama-3-8B-Instruct"),
         ("criscar22/Qwen2.5-Coder-7B-Instruct-abliterated", "Qwen/Qwen2.5-Coder-7B-Instruct"),
         ("QuixiAI/Llama-3-8B-Instruct-abliterated-v2", "NousResearch/Meta-Llama-3-8B-Instruct")]
K = 3


def svals(D):
    s = np.linalg.svd(D.astype(np.float64), compute_uv=False)
    return s / max(s[0], 1e-30)


def main(out):
    res = []
    for child, parent in PAIRS:
        rc, rp = R.Remote(child), R.Remote(parent)
        L = rc.config["num_hidden_layers"]
        layers = sorted({int(round((j + 0.5) * L / K - 0.5)) for j in range(K)})
        rec = {"child": child, "parent": parent, "layers": {}}
        for i in layers:
            a, b = R.build(rc, layers=[i])["layers"][i], R.build(rp, layers=[i])["layers"][i]
            row = {}
            for n in ("q", "k", "v", "o", "gate", "up", "down"):
                D = a[n] - b[n]
                rel = float(np.linalg.norm(D) / np.linalg.norm(b[n]))
                s = svals(D) if rel > 1e-6 else np.zeros(3)
                row[n] = {"rel": rel, "s2_over_s1": float(s[1]) if len(s) > 1 else 0.0}
            rec["layers"][i] = row
            print(f"{child.split('/')[0]:9s} layer {i:2d} " + "  ".join(
                f"{n}:{row[n]['rel']:.1e}/{row[n]['s2_over_s1']:.2f}" for n in row), flush=True)
        Ea, Eb = R.build(rc, layers=[])["E"], R.build(rp, layers=[])["E"]
        rec["embed_rel"] = float(np.linalg.norm(Ea - Eb) / np.linalg.norm(Eb))
        print(f"{child.split('/')[0]:9s} embedding rows: rel edit {rec['embed_rel']:.1e}", flush=True)
        res.append(rec)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
