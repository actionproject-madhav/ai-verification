"""Abliteration census: real Hub models, forensics from weights alone (no prompts, no parent model).

For each sampled layer l, W_l = [o_proj, down_proj] (everything layer l writes to the residual stream):
  isolation_l = lambda_1 / lambda_2 of W_l W_l^T (smallest two eigenvalues)   ~1 normally, << 1 if one direction was cut
  r_l = the eigenvector of lambda_1
Embedding rows likewise (isolation_E, r_E). Abliteration cuts the SAME direction from every writer, so the cut
directions agree across layers: |r_l . r_l'| ~ 1. Verdict 'ablated' when at least two writers show isolation < 0.1
and their cut directions agree (|cos| > 0.9). All of it is invariant to rotations and permutations.

Usage: python exp_census.py <out.json>
"""
import json, sys, time, traceback
import numpy as np
from lineage_check import remote as R

ABLATED = ["failspy/Meta-Llama-3-8B-Instruct-abliterated-v3", "QuixiAI/Llama-3-8B-Instruct-abliterated-v2",
           "mlabonne/Daredevil-8B-abliterated", "mlabonne/NeuralDaredevil-8B-abliterated",
           "byroneverson/Yi-1.5-9B-Chat-abliterated", "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2",
           "criscar22/Qwen2.5-Coder-7B-Instruct-abliterated", "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
           "huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2"]
CONTROLS = ["NousResearch/Meta-Llama-3-8B-Instruct", "NousResearch/Hermes-2-Pro-Llama-3-8B", "01-ai/Yi-1.5-9B-Chat",
            "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct"]
K = 3


def cut(W):
    ev, vec = np.linalg.eigh(W @ W.T)
    return float(ev[0] / max(ev[1], 1e-30)), float(ev[0] / np.median(ev)), vec[:, 0]


def analyse(src):
    meta = R.build(src, layers=[])
    L = meta["L"]
    layers = sorted({int(round((j + 0.5) * L / K - 0.5)) for j in range(K)})
    out, dirs = {"layers": {}, "kind": meta["kind"], "d": meta["d"], "L": L}, []
    for i in layers:
        l = R.build(src, layers=[i])["layers"][i]
        W = np.concatenate([l["o"], l["down"]], 1).astype(np.float64)
        del l
        iso, rel, r = cut(W / np.linalg.norm(W))
        out["layers"][i] = {"isolation": iso, "rel": rel}
        if iso < 0.1:
            dirs.append(r)
    E = meta["E"].astype(np.float64)
    iso, rel, r = cut(E.T / np.linalg.norm(E))
    out["embed"] = {"isolation": iso, "rel": rel}
    if iso < 0.1:
        dirs.append(r)
    cos = [abs(float(dirs[a] @ dirs[b])) for a in range(len(dirs)) for b in range(a + 1, len(dirs))]
    out["n_cut"] = len(dirs)
    out["agreement"] = float(np.min(cos)) if cos else None
    out["verdict"] = "ablated" if len(dirs) >= 2 and cos and min(cos) > 0.9 else "clean"
    return out


def main(out):
    res = json.load(open(out)) if __import__("os").path.exists(out) else []
    done = {r["model"] for r in res}
    for truth, repos in (("ablated", ABLATED), ("control", CONTROLS)):
        for repo in repos:
            if repo in done:
                continue
            t0 = time.time()
            try:
                a = analyse(R.Remote(repo))
                rec = {"model": repo, "truth": truth, **a,
                       "seconds": round(time.time() - t0)}
            except Exception as e:
                rec = {"model": repo, "truth": truth, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                traceback.print_exc()
            res.append(rec)
            json.dump(res, open(out, "w"), indent=1)
            if "error" in rec:
                print(f"{repo:52s} ERROR {rec['error']}", flush=True)
            else:
                iso = " ".join(f"{v['isolation']:.0e}" for v in rec["layers"].values())
                print(f"{repo:52s} [{truth:7s}] -> {rec['verdict']:7s} layer isolation {iso} | embed {rec['embed']['isolation']:.0e}"
                      f" | agree {rec['agreement'] if rec['agreement'] is None else round(rec['agreement'], 3)} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
