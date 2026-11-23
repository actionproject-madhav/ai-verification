"""Does the verdict depend on the audit's random draw? 10 nonces x {512, 64} tokens x {all layers, 2 nonce-chosen layers}.

Usage: python exp_stability.py <out.json>
"""
import glob, hashlib, json, os, sys
import numpy as np
import torch
from lineage_check import fingerprint as fp4
from lineage_check import remote as R
from lineage_check import paths

torch.set_grad_enabled(False)
H = paths.HUB
NONCES = [f"stability-{k}" for k in range(10)]


def snap(repo):
    base = f"{H}/models--{repo.replace('/', '--')}"
    ref = os.path.join(base, "refs", "main")
    return os.path.join(base, "snapshots", open(ref).read().strip()) if os.path.exists(ref) else sorted(glob.glob(f"{base}/snapshots/*"))[0]


def layer_pick(L, nonce, k=2):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("L" + nonce).encode()).digest()[:8], "little"))
    return sorted(rng.choice(L, size=k, replace=False).tolist())


def fps_for(src_fn, vocab, label):
    m = R.build(src_fn())
    out = {}
    for nonce in NONCES:
        for n in (512, 64):
            strings = fp4.select_strings(vocab, nonce, n)
            rows = fp4.rows_for(m, vocab, strings)
            keep = [r for r in rows if r is not None]
            fp = fp4.fingerprint(m, keep)
            out[(nonce, n)] = (fp, [r is not None for r in rows])
    del m
    print("fingerprinted", label, flush=True)
    return out


def dist(a, b, nonce, L, sub_layers):
    fa, pa = a
    fb, pb = b
    assert pa == pb
    la = {i: fa["layers"][i] for i in (sub_layers or fa["layers"])}
    lb = {i: fb["layers"][i] for i in (sub_layers or fb["layers"])}
    r = fp4.compare({"EE": fa["EE"], "EU": fa["EU"], "layers": la}, {"EE": fb["EE"], "EU": fb["EU"], "layers": lb})
    return float(np.mean([r["qk"], r["ov"]]))


def main(out):
    qv = R.Remote("Qwen/Qwen2.5-0.5B").vocab
    pv = R.Remote("EleutherAI/pythia-160m").vocab
    Q = {k: fps_for(lambda r=r: R.LocalST(snap(r), repo=r), qv, k) for k, r in [
        ("base", "Qwen/Qwen2.5-0.5B"), ("Instruct", "Qwen/Qwen2.5-0.5B-Instruct"), ("Coder", "Qwen/Qwen2.5-Coder-0.5B"),
        ("abliterated 4-bit", "mlx-community/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-4bit"), ("Qwen2 (prev gen)", "Qwen/Qwen2-0.5B")]}
    def pythia(r):
        p = snap(r)
        return R.LocalST(p, repo=r) if os.path.exists(os.path.join(p, "model.safetensors")) else R.LocalBin(p, label=r)
    P = {k: fps_for(lambda r=r: pythia(r), pv, k) for k, r in [
        ("main", "EleutherAI/pythia-160m")] + [(f"seed{s}", f"EleutherAI/pythia-160m-seed{s}") for s in range(1, 10)]}
    res = {}
    for setting in ("512 tokens, all layers", "64 tokens, all layers", "512 tokens, 2 layers", "64 tokens, 2 layers"):
        n = 512 if setting.startswith("512") else 64
        two = "2 layers" in setting
        rows = {}
        for lab, fam, a, b, L in [("Qwen Instruct (derivative)", Q, "base", "Instruct", 24),
                                  ("abliterated 4-bit (derivative)", Q, "base", "abliterated 4-bit", 24),
                                  ("Coder +5.5T (heavy)", Q, "base", "Coder", 24),
                                  ("Qwen2 (previous generation)", Q, "base", "Qwen2 (prev gen)", 24)] + \
                                 [(f"Pythia seed{s} (independent)", P, "main", f"seed{s}", 12) for s in range(1, 10)]:
            vals = [dist(fam[a][(nc, n)], fam[b][(nc, n)], nc, L, layer_pick(L, nc) if two else None) for nc in NONCES]
            rows[lab] = {"min": min(vals), "max": max(vals), "mean": float(np.mean(vals))}
        der = max(v["max"] for k, v in rows.items() if "(derivative)" in k)
        ind = min(v["min"] for k, v in rows.items() if "independent" in k)
        res[setting] = {"rows": rows, "worst_derivative": der, "closest_independent": ind, "margin": ind / der}
        print(f"{setting:24s} worst derivative {der:.4f}   closest independent {ind:.4f}   margin x{ind / der:.0f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
