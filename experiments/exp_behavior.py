"""Does a behavioural check still link the attacked models to their parent?

The round-4 attacker keeps the parent's outputs and changes its internals, so it should stay close in behaviour.
For each model: mean KL(parent || model) per token and top-1 agreement with the parent, on 64 windows of 128 tokens
of WikiText-103 validation text, which no attack trained on.

Usage: python exp_behavior.py <out.json>
"""
import json, os, sys
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
import exp_adaptive as A
from lineage_check import paths

torch.set_grad_enabled(False)
F = torch.nn.functional
DEV = A.DEV
SAVE = paths.ADAPTIVE
MODELS = [("parent (SmolLM2-135M)", "HuggingFaceTB/SmolLM2-135M"),
          ("derivative: SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"),
          ("honest fine-tune, narrow data", f"{SAVE}/ft"),
          ("honest fine-tune, broad data", f"{SAVE}/r3-honest"),
          ("round 1: attack on v3", f"{SAVE}/sec10"),
          ("round 2: attack on v4", f"{SAVE}/v4"),
          ("round 2: attack on both", f"{SAVE}/both"),
          ("round 4: distil + attack both, lambda=3", f"{SAVE}/r4-distill-lam3"),
          ("round 4: distil + attack both, lambda=1", f"{SAVE}/r4-distill-lam1"),
          ("independent: SmolLM-135M v1", "HuggingFaceTB/SmolLM-135M")]


def probe():
    p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet", repo_type="dataset")
    text = "\n".join(t for t in pq.read_table(p, columns=["text"]).column("text").to_pylist() if t.strip())
    ids = torch.tensor(A.tok(text)["input_ids"])
    n = ids.numel() // 129
    W = ids[: n * 129].view(n, 129)
    return W[torch.randperm(n, generator=torch.Generator().manual_seed(23))[:64], :128]


def main(out):
    X = probe().to(DEV)
    parent = A.load(MODELS[0][1]).eval()
    P = torch.cat([F.log_softmax(parent(X[i:i + 16]).logits.float(), -1).cpu() for i in range(0, 64, 16)])
    del parent
    res = []
    for lab, path in MODELS:
        m = A.load(path).eval()
        Q = torch.cat([F.log_softmax(m(X[i:i + 16]).logits.float(), -1).cpu() for i in range(0, 64, 16)])
        kl = float((P.exp() * (P - Q)).sum(-1).mean())
        top1 = float((P.argmax(-1) == Q.argmax(-1)).float().mean())
        res.append({"model": lab, "kl_to_parent": kl, "top1_agreement": top1})
        print(f"{lab:44s} KL {kl:.3f}  top-1 agreement {top1:.3f}", flush=True)
        del m, Q
        if DEV == "mps":
            torch.mps.empty_cache()
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
