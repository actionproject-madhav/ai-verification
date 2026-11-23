"""Frontier coverage: mixture-of-experts models (Qwen3-30B-A3B, OLMoE-1B-7B), 2 sampled layers each.

Also an exact symmetry attack specific to MoE: shuffle the experts (with their router rows), add one vector to every
router row (changes no routing decision), and permute / rescale neurons inside each expert. And the refusal-ablation
signature on a public abliterated MoE release (writers = attention output plus every expert's down projection).

Usage: python exp_moe.py <out.json>
"""
import copy, gc, json, sys
import numpy as np
import torch
from lineage_check import fingerprint as fp4, forensic
from lineage_check import remote as R
from lineage_check import scan

torch.set_grad_enabled(False)
NONCE, N_TOK, K = "lineage-audit-2026-11-18", 512, 2


def moe_symmetry_attack(m, seed=0):
    """Exact for routed MoE: expert order, router offset, neuron order and up/down scale inside each expert."""
    m = copy.deepcopy(m)
    rng = np.random.default_rng(seed)
    for l in m["layers"].values():
        P = rng.permutation(len(l["experts"]))
        l["experts"] = [l["experts"][p] for p in P]
        l["router"] = l["router"][P] + rng.normal(0, 1.0, (1, l["router"].shape[1])).astype(np.float32)
        new = []
        for g, u, dn in l["experts"]:
            pi = rng.permutation(u.shape[0])
            c = np.exp(rng.uniform(-1, 1, u.shape[0])).astype(np.float32)
            new.append((g[pi], u[pi] * c[:, None], dn[:, pi] / c[None, :]))
        l["experts"] = new
    return m


def fp_of(m, vocab, strings, label):
    rows = fp4.rows_for(m, vocab, strings)
    present = [r is not None for r in rows]
    fp = fp4.fingerprint(m, [r for r in rows if r is not None])
    fp.update(label=label, strings=strings, present=present, L=m["L"], d=m["d"], kind=m["kind"])
    return fp


def moe_forensic(m):
    W, Rd = [], []
    for l in m["layers"].values():
        W.append(np.concatenate([l["o"]] + [dn for _, _, dn in l["experts"]], 1))
        g1, g2 = l["g1"][None, :], l["g2"][None, :]
        Rd.append(np.concatenate([l["q"] * g1, l["k"] * g1, l["v"] * g1, l["router"] * g2] +
                                 [g * g2 for g, _, _ in l["experts"][:16]] + [u * g2 for _, u, _ in l["experts"][:16]], 0))
    return forensic.verdict(W + [m["E"].T], Rd + [None])


def main(out):
    res = {}
    for fam, anchor, queries in [
        ("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-Base",
         [("post-trained (Qwen3-30B-A3B)", "Qwen/Qwen3-30B-A3B", "derivative"),
          ("Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", "derivative"),
          ("Thinking-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507", "derivative"),
          ("Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "derivative"),
          ("abliterated Instruct-2507 (huihui-ai)", "huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated", "derivative")]),
        ("olmoe-1b-7b", "allenai/OLMoE-1B-7B-0924",
         [("0924-Instruct", "allenai/OLMoE-1B-7B-0924-Instruct", "derivative"),
          ("0125 (separate run)", "allenai/OLMoE-1B-7B-0125", "other run"),
          ("0125-Instruct", "allenai/OLMoE-1B-7B-0125-Instruct", "other run")])]:
        src = R.Remote(anchor)
        vocab = src.vocab
        strings = fp4.select_strings(vocab, NONCE, N_TOK)
        fa = scan.fingerprint_source(src, strings, K, label=f"moe/{fam}/anchor")
        rec = []
        # exact MoE symmetry attack on the anchor itself
        L = src.config["num_hidden_layers"]
        layers = sorted({int(round((j + 0.5) * L / K - 0.5)) for j in range(K)})
        base = R.build(src, layers=layers)
        fb = fp_of(moe_symmetry_attack(base), vocab, strings, "attacked")
        r = scan.compare(fa, fb)
        rec.append({"query": "anchor after exact MoE symmetry attack", "role": "attack", "attn": r["attn"], "d": r["d"],
                    "router": r.get("router")})
        print(f"[{fam}] exact MoE symmetry attack on the anchor: attn {r['attn']:.2e}  all {r['d']:.2e}", flush=True)
        del base
        gc.collect()
        for label, repo, role in queries:
            try:
                qs = R.Remote(repo)
                fq = scan.fingerprint_source(qs, strings, K, label=f"moe/{fam}/{label}", vocab=qs.vocab)
                r = scan.compare(fa, fq)
                row = {"query": label, "repo": repo, "role": role, "attn": r["attn"], "d": r["d"], "router": r.get("router"),
                       "coverage": r["coverage"]}
                if "abliterated" in label or role == "derivative":
                    mq = R.build(qs, layers=layers)
                    ab, k, agree, ks, rr = moe_forensic(mq)
                    row.update(ablation_flag=ab, dead=ks, read=rr)
                    del mq
                    gc.collect()
                rec.append(row)
                print(f"[{fam}] {label:40s} attn {r['attn']:.4f}  all {r['d']:.4f}  router {r.get('router', float('nan')):.3f}"
                      f"{'  ABLATED' if row.get('ablation_flag') else ''}", flush=True)
            except Exception as e:
                rec.append({"query": label, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"[{fam}] {label}: ERROR {type(e).__name__}: {str(e)[:150]}", flush=True)
        res[fam] = rec
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
