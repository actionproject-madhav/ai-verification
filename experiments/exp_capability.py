"""What does evading the fingerprints cost in general capability (not just in-domain loss)?

Models: SmolLM2-135M parent, an honest fine-tune (ft), the v3 attack (sec10), the v4 attack (v4), the joint
attack (both), and an independently trained model (SmolLM-135M v1) for reference.
Metrics: LAMBADA (OpenAI version) last-word accuracy, greedy, first 1,000 examples; TinyStories validation
cross-entropy (out of domain for the WikiText fine-tunes), 400 windows of 128 tokens.

Usage: python exp_capability.py <out.json>
"""
import json, os, sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from lineage_check import paths

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ZOO = paths.ADAPTIVE
tok = AutoTokenizer.from_pretrained(os.environ.get("RT_MODEL", "HuggingFaceTB/SmolLM2-135M"))
MODELS = [("parent (SmolLM2-135M)", "HuggingFaceTB/SmolLM2-135M"), ("honest fine-tune", f"{ZOO}/ft"),
          ("attack on v3", f"{ZOO}/sec10"), ("attack on v4", f"{ZOO}/v4"), ("attack on both", f"{ZOO}/both"),
          ("independent (SmolLM-135M v1)", "HuggingFaceTB/SmolLM-135M")]

lam = [json.loads(l) for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))][:1000]
ts = open(hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")).read()
ts_ids = torch.tensor(tok(ts[:2_000_000])["input_ids"])
n = min(400, ts_ids.numel() // 129)
TS = ts_ids[: n * 129].view(n, 129)


def lambada(m):
    ok = 0
    for ex in lam:
        text = ex["text"]
        ctx, word = text.rsplit(" ", 1)
        c = tok(ctx, add_special_tokens=False)["input_ids"]
        w = tok(" " + word, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([c + w], device=DEV)
        pred = m(ids).logits[0, len(c) - 1: len(c) + len(w) - 1].argmax(-1).tolist()
        ok += pred == w
    return ok / len(lam)


def tinystories(m):
    ces = []
    for i in range(0, TS.shape[0], 16):
        b = TS[i:i + 16].to(DEV)
        ces.append(float(torch.nn.functional.cross_entropy(m(b[:, :-1]).logits.flatten(0, 1), b[:, 1:].flatten())))
    return float(np.mean(ces))


def main(out):
    res = []
    for name, path in MODELS:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
        r = {"model": name, "lambada_acc": lambada(m), "tinystories_ce": tinystories(m)}
        res.append(r)
        print(f"{name:32s} LAMBADA {r['lambada_acc']:.3f}   TinyStories CE {r['tinystories_ce']:.3f}", flush=True)
        del m
        if DEV == "mps":
            torch.mps.empty_cache()
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
