"""How far apart are unrelated models under the design-free fingerprint? Every pair in a set of base models.

The traffic light flags a model when its closest part is under 0.5 from a listed model. This measures the margin
on models nobody claims are related: each pair of the bases below, on tokens and layers drawn from a nonce.
Pairs within one family (two sizes of one release, or two versions) are kept and labelled, so they can be read
separately.

Usage: python exp_universal_cross.py <out.json> [nonce]
"""
import itertools, json, os, shutil, sys, time
import torch
from lineage_check import light as LT, remote as R

torch.set_grad_enabled(False)
MODELS = [  # (repo, family): the same family means the same developer and model line
    ("unsloth/Llama-3.1-8B", "llama-3"), ("marin-community/marin-8b-base", "marin"),
    ("Qwen/Qwen2.5-7B", "qwen2.5"), ("Qwen/Qwen2.5-1.5B", "qwen2.5"), ("Qwen/Qwen3-8B-Base", "qwen3"),
    ("mistralai/Mistral-7B-v0.3", "mistral"), ("allenai/OLMo-2-1124-7B", "olmo-2"),
    ("HuggingFaceTB/SmolLM2-1.7B", "smollm2"), ("HuggingFaceTB/SmolLM-1.7B", "smollm"),
    ("HuggingFaceTB/SmolLM2-360M", "smollm2"), ("HuggingFaceTB/SmolLM-360M", "smollm"),
    ("HuggingFaceTB/SmolLM2-135M", "smollm2"), ("HuggingFaceTB/SmolLM-135M", "smollm"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama"), ("EleutherAI/pythia-1.4b", "pythia"),
    ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-v4"), ("zai-org/GLM-5.3-Flash", "glm-flash"),
]


def main(out, nonce="universal-cross-2026-11"):
    srcs, rows, cache = {}, [], {}
    for repo, _ in MODELS:
        try:
            srcs[repo] = R.Remote(repo)
        except Exception as e:
            print("skip", repo, type(e).__name__, str(e)[:100], flush=True)
    fam = dict(MODELS)
    for a, b in itertools.combinations(srcs, 2):
        t0 = time.time()
        row = {"a": a, "b": b, "same_family": fam[a] == fam[b]}
        try:
            d = LT.distances(srcs[b], srcs[a], nonce, cache=cache)
            row.update(closest_part=None if d is None else round(d[0], 4), all_parts=None if d is None else round(d[1], 4))
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {str(e)[:120]}")
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        print(f"{a[:34]:34s} {b[:34]:34s} {row.get('closest_part')} {row.get('all_parts')} {row.get('error', '')}", flush=True)
        json.dump({"nonce": nonce, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
