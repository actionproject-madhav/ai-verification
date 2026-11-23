"""The two forms of a refusal cut, on five small public models.

Plain abliteration leaves W - r r'W. Heretic makes the same cut on the matrix with its rows scaled to unit length
and scales them back. For every changed writing matrix this script reports the cosine between what left the matrix
along the leading direction of its change and what the base held there, in both forms. `universal.edit_directions`
keeps the larger of the two; this shows which form each tool uses.

Usage: python exp_cut_forms.py <out.json>
"""
import json, sys
import torch
from lineage_check import light as LT, remote as R, universal as U

torch.set_grad_enabled(False)
NONCE = "cut-forms-2026-11"
PAIRS = [("unsloth/Qwen2.5-0.5B-Instruct", "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3", "abliterated"),
         ("Qwen/Qwen3-1.7B", "huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2", "abliterated"),
         ("Qwen/Qwen3-1.7B", "mlabonne/Qwen3-1.7B-abliterated", "abliterated"),
         ("Qwen/Qwen2.5-1.5B-Instruct", "SonGT1/Qwen2.5-1.5B-Instruct-abliterated", "abliterated"),
         ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "saidutta69/SmolLM2-1.7B-Instruct-heretic", "heretic")]


def cosine(a, b):
    return abs(float(a @ b)) / max(float(a.norm() * b.norm()), 1e-30)


def main(out):
    rows = []
    for base, model, tool in PAIRS:
        anchor, src = R.Remote(base), R.Remote(model)
        ls, la = U.layout(src)[2], U.layout(anchor)[2]
        plain, scaled, share = [], [], []
        for i, j in zip(LT.layers_for(src, NONCE, LT.CUT_LAYERS), LT.layers_for(anchor, NONCE, LT.CUT_LAYERS)):
            for n, kind in ls[i]["mats"]:
                if kind != "w" or n not in dict(la[j]["mats"]):
                    continue
                W = U.matrix(anchor, n)
                D = U.matrix(src, n) - W
                if float(D.norm()) <= 1e-6 * float(W.norm()):
                    continue                                       # this matrix was left alone
                lead, s = U._leading(D)
                u, rows_sq = lead[:, 0], (W ** 2).sum(1).clamp_min(1e-30)
                gone = u @ D
                plain.append(round(cosine(gone, u @ W), 3))
                scaled.append(round(cosine(gone, (u / rows_sq) @ W), 3))
                share.append(round(s, 3))
        rows.append({"base": base, "model": model, "name_says": tool, "matrices_changed": len(plain), "rank_one_share": share,
                     "own_content_plain": plain, "own_content_rows_scaled": scaled})
        print(f"{model:52s} plain {min(plain):.2f} to {max(plain):.2f}   rows scaled {min(scaled):.2f} to {max(scaled):.2f}", flush=True)
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
