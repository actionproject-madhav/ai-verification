"""Forensics without the parent: does a model carry the weight signature of refusal-direction ablation?

Abliteration (weight orthogonalisation) replaces every residual writer W (o_proj, down_proj, embedding) by
(I - r r^T) W. That leaves a residual direction that nothing ever writes to, while readers (q, k, v, gate, up)
still read it. Residual padding also leaves dead directions, but nothing reads those either.

Statistic, per model (rotation- and permutation-invariant):
  Cw = sum over writers of W W^T / ||W||_F^2     (d x d)        smallest eigenpair (lambda_min, r)
  write gap  = lambda_min / median eigenvalue of Cw              ~1e-6 to 1e-3 when a direction was ablated
  read ratio = (r^T Cr r) / mean eigenvalue of Cr                Cr = sum over readers of W^T W / ||W||^2
                                                                 ~1 for ablation, ~0 for padding

Usage: python exp_forensics.py <out.json> [7b]
"""
import json, sys
import numpy as np
import torch
from lineage_check import edits as attacks, canon
from lineage_check import remote as R
from lineage_check import paths

torch.set_grad_enabled(False)


def stats(m):
    d = m["E"].shape[1]
    Cw = np.zeros((d, d))
    Cr = np.zeros((d, d))
    E = m["E"].astype(np.float64)
    Cw += (E.T @ E) / np.sum(E * E)
    for l in m["layers"].values():
        for n in ("o", "down"):
            W = l[n].astype(np.float64)
            Cw += (W @ W.T) / np.sum(W * W)
        g1, g2 = l["g1"].astype(np.float64), l["g2"].astype(np.float64)
        for n, g in (("q", g1), ("k", g1), ("v", g1), ("gate", g2), ("up", g2)):
            if l.get(n) is not None:
                W = l[n].astype(np.float64) * g[None, :]
                Cr += (W.T @ W) / np.sum(W * W)
    ev, vec = np.linalg.eigh(Cw)
    r = vec[:, 0]
    return {"write_gap": float(ev[0] / np.median(ev)), "second_gap": float(ev[1] / np.median(ev)),
            "read_ratio": float(r @ Cr @ r / np.mean(np.linalg.eigvalsh(Cr))), "n_layers": len(m["layers"])}


def main(out, big):
    res = []
    H = paths.HUB
    rem = lambda repo, layers="all": R.build(R.Remote(repo), layers=layers)
    if not big:
        base_c = canon.load(sorted(__import__("glob").glob(f"{H}/models--Qwen--Qwen2.5-0.5B/snapshots/*"))[0])
        abl4 = "mlx-community/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-4bit"
        abl4_c = canon.load(sorted(__import__("glob").glob(f"{H}/models--{abl4.replace('/', '--')}/snapshots/*"))[0])
        cases = [("Qwen2.5-0.5B base", "control", lambda: rem("Qwen/Qwen2.5-0.5B")),
                 ("Qwen2.5-0.5B-Instruct", "control", lambda: rem("Qwen/Qwen2.5-0.5B-Instruct")),
                 ("Qwen2.5-0.5B-Instruct AWQ 4-bit", "control", lambda: rem("Qwen/Qwen2.5-0.5B-Instruct-AWQ")),
                 ("Qwen2.5-Coder-0.5B", "control", lambda: rem("Qwen/Qwen2.5-Coder-0.5B")),
                 ("Qwen2-0.5B", "control", lambda: rem("Qwen/Qwen2-0.5B")),
                 ("base + synthetic rank-1 ablation", "ablated", lambda: R.from_canon(attacks.rank1_edit(base_c))),
                 ("Josiefied abliterated-v1 (real)", "ablated",
                  lambda: rem("Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1")),
                 ("  ... MLX 4-bit re-release (real)", "ablated", lambda: rem(abl4)),
                 ("  ... 4-bit + all six reparameterisations", "ablated", lambda: R.from_canon(attacks.obfuscate(abl4_c, 4))),
                 ("base + residual padding 896->960 (no ablation)", "padded",
                  lambda: R.from_canon(attacks.widen_residual(base_c, 64)))]
    else:
        cases = [("Qwen2.5-7B-Instruct", "control", lambda: rem("Qwen/Qwen2.5-7B-Instruct", 4)),
                 ("Qwen2.5-7B-Instruct AWQ", "control", lambda: rem("Qwen/Qwen2.5-7B-Instruct-AWQ", 4)),
                 ("huihui-ai abliterated-v2 (real)", "ablated", lambda: rem("huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2", 4)),
                 ("Qwen2.5-72B-Instruct", "control", lambda: rem("Qwen/Qwen2.5-72B-Instruct", 2)),
                 ("huihui-ai 72B abliterated (real)", "ablated", lambda: rem("huihui-ai/Qwen2.5-72B-Instruct-abliterated", 2))]
    skip = set(sys.argv[3:])
    for label, kind, make in cases:
        if any(label.startswith(x) for x in skip):
            continue
        r = {"model": label, "truth": kind, **stats(make())}
        res.append(r)
        print(f"{label:48s} [{kind:7s}] write gap {r['write_gap']:.2e}  next {r['second_gap']:.2e}  read ratio {r['read_ratio']:.2f}"
              f"  ({r['n_layers']} layers)", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "7b")
