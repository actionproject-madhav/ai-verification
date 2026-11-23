"""The cut test on a few real pairs, over several nonces: refusal-removed models against the known base, and honest
versions of the same bases. The base is the registry's entry, which is not always the model the edit started from
(an edited instruct model is compared with the plain base).

Usage: python exp_cut_pairs.py <out.json> [nonces]
"""
import json, sys, time
import torch
from lineage_check import light as LT, remote as R, universal as U

torch.set_grad_enabled(False)
PAIRS = [  # known base, model, what it is
    ("unsloth/Llama-3.1-8B", "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", "refusals removed"),
    ("Qwen/Qwen2.5-7B", "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2", "refusals removed"),
    ("Qwen/Qwen3-8B-Base", "DreamFast/qwen3-8b-heretic", "refusals removed, newer tool"),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "instruct version"),
    ("unsloth/Llama-3.1-8B", "NousResearch/Hermes-3-Llama-3.1-8B", "fine-tune"),
    ("unsloth/Llama-3.1-8B", "unsloth/Llama-3.1-8B-Instruct", "instruct version"),
    ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B", "instruct version"),
]


def main(out, nonces=4):
    rows = []
    for base, model, what in PAIRS:
        t0 = time.time()
        a, q = R.Remote(base), R.Remote(model)
        cut, aligned = [], []
        for i in range(int(nonces)):
            nonce = f"cut-pairs-{i}"
            r = U.cut_layers(q, a, LT.layers_for(q, nonce, LT.CUT_LAYERS), LT.layers_for(a, nonce, LT.CUT_LAYERS))
            cut.append(r["cut"])
            aligned.append(r["aligned"])
        d = LT.distances(q, a, "cut-pairs-0")
        rows.append({"base": base, "model": model, "what": what, "cut_layers_per_nonce": cut, "layers_read": LT.CUT_LAYERS,
                     "red": [c >= LT.CUT_MIN for c in cut], "aligned": aligned, "all_parts": round(d[1], 4),
                     "seconds": round(time.time() - t0)})
        print(rows[-1], flush=True)
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
